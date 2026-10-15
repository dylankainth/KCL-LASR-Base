"""Minimal URDF forward/inverse kinematics (numpy only).

Used by both the scripted expert and the learned policy so that they act in
Cartesian space (end-effector targets) instead of raw joint space.
"""

import xml.etree.ElementTree as ET

import numpy as np


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _transform(rot, trans):
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = trans
    return t


class Chain:
    """Serial chain from `base` link to `tip` link, parsed from a URDF string."""

    def __init__(self, urdf_xml: str, base: str, tip: str):
        root = ET.fromstring(urdf_xml.strip())
        by_child = {}
        for joint in root.findall("joint"):
            by_child[joint.find("child").get("link")] = joint

        path = []
        link = tip
        while link != base:
            if link not in by_child:
                raise ValueError(f"No path from {base} to {tip} (stuck at {link})")
            joint = by_child[link]
            path.append(joint)
            link = joint.find("parent").get("link")
        path.reverse()

        self.base, self.tip = base, tip
        self.joint_names = []  # movable joints, in chain order
        self._elems = []  # (origin 4x4, movable index or None, axis, type)
        self.lower, self.upper = [], []
        for joint in path:
            origin = joint.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
                rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
            t_origin = _transform(rpy_to_matrix(*rpy), xyz)
            jtype = joint.get("type")
            if jtype in ("revolute", "prismatic", "continuous"):
                axis_el = joint.find("axis")
                axis = (
                    np.array([float(v) for v in axis_el.get("xyz").split()])
                    if axis_el is not None
                    else np.array([1.0, 0, 0])
                )
                limit = joint.find("limit")
                lo = float(limit.get("lower", -np.pi)) if limit is not None else -np.pi
                hi = float(limit.get("upper", np.pi)) if limit is not None else np.pi
                self._elems.append((t_origin, len(self.joint_names), axis, jtype))
                self.joint_names.append(joint.get("name"))
                self.lower.append(lo)
                self.upper.append(hi)
            else:
                self._elems.append((t_origin, None, None, jtype))
        self.lower = np.array(self.lower)
        self.upper = np.array(self.upper)

    @property
    def n(self):
        return len(self.joint_names)

    def fk(self, q) -> np.ndarray:
        """4x4 pose of the tip in the base frame for the movable-joint vector q."""
        t = np.eye(4)
        for origin, idx, axis, jtype in self._elems:
            t = t @ origin
            if idx is not None:
                if jtype == "prismatic":
                    t = t @ _transform(np.eye(3), axis * q[idx])
                else:
                    t = t @ _transform(axis_angle_matrix(axis, q[idx]), np.zeros(3))
        return t

    def jacobian(self, q, eps=1e-6) -> np.ndarray:
        """Numeric 6xN geometric Jacobian (linear on top, angular below)."""
        q = np.asarray(q, dtype=float)
        t0 = self.fk(q)
        jac = np.zeros((6, self.n))
        for i in range(self.n):
            dq = q.copy()
            dq[i] += eps
            t1 = self.fk(dq)
            jac[:3, i] = (t1[:3, 3] - t0[:3, 3]) / eps
            # world-frame angular velocity from R1 R0^T
            d_r = t1[:3, :3] @ t0[:3, :3].T
            w = 0.5 * np.array(
                [d_r[2, 1] - d_r[1, 2], d_r[0, 2] - d_r[2, 0], d_r[1, 0] - d_r[0, 1]]
            )
            jac[3:, i] = w / eps
        return jac


def rotation_error(r_cur, r_tgt):
    """Small-angle rotation vector taking r_cur to r_tgt (world frame)."""
    return 0.5 * sum(np.cross(r_cur[:, i], r_tgt[:, i]) for i in range(3))


def rotation_from_axes(approach_local, finger_local, approach_world, finger_world):
    """Rotation R with R@approach_local = approach_world, R@finger_local ~ finger_world."""
    a_l = np.asarray(approach_local, float)
    a_l = a_l / np.linalg.norm(a_l)
    f_l = np.asarray(finger_local, float)
    f_l = f_l - a_l * (a_l @ f_l)
    f_l = f_l / np.linalg.norm(f_l)
    a_w = np.asarray(approach_world, float)
    a_w = a_w / np.linalg.norm(a_w)
    f_w = np.asarray(finger_world, float)
    f_w = f_w - a_w * (a_w @ f_w)
    f_w = f_w / np.linalg.norm(f_w)
    local = np.stack([a_l, f_l, np.cross(a_l, f_l)], axis=1)
    world = np.stack([a_w, f_w, np.cross(a_w, f_w)], axis=1)
    return world @ local.T


def solve_ik(
    chain: Chain,
    q0,
    target_pos,
    target_rot=None,
    q_pref=None,
    pref_gain=0.2,
    rot_weight=0.5,
    damping=0.05,
    max_iters=200,
    pos_tol=1e-3,
    rot_tol=2e-2,
    free=None,
):
    """Damped-least-squares IK with joint limits and a posture preference.

    `free` is an optional boolean mask of joints that may move (others stay at q0).
    Returns (q, converged).
    """
    q = np.array(q0, dtype=float)
    free = np.ones(chain.n, bool) if free is None else np.asarray(free, bool)
    q_pref = q.copy() if q_pref is None else np.asarray(q_pref, float)
    for _ in range(max_iters):
        t = chain.fk(q)
        e_pos = np.asarray(target_pos) - t[:3, 3]
        if target_rot is not None:
            e_rot = rotation_error(t[:3, :3], target_rot)
        else:
            e_rot = np.zeros(3)
        if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
            return q, True
        jac = chain.jacobian(q)[:, free]
        if target_rot is None:
            jac = jac[:3]
            err = e_pos
        else:
            jac = np.vstack([jac[:3], rot_weight * jac[3:]])
            err = np.concatenate([e_pos, rot_weight * e_rot])
        jjt = jac @ jac.T + damping**2 * np.eye(jac.shape[0])
        j_pinv = jac.T @ np.linalg.inv(jjt)
        dq = j_pinv @ err
        null = np.eye(jac.shape[1]) - j_pinv @ jac
        dq += null @ (pref_gain * (q_pref[free] - q[free]))
        step = np.max(np.abs(dq))
        if step > 0.2:
            dq *= 0.2 / step
        q_new = q.copy()
        q_new[free] += dq
        q = np.clip(q_new, chain.lower, chain.upper)
    t = chain.fk(q)
    ok = (
        np.linalg.norm(np.asarray(target_pos) - t[:3, 3]) < 5 * pos_tol
        and (
            target_rot is None
            or np.linalg.norm(rotation_error(t[:3, :3], target_rot)) < 5 * rot_tol
        )
    )
    return q, ok
