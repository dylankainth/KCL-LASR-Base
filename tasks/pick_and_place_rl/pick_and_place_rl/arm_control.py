"""Cartesian arm control on top of TiagoScene (top-down grasps, torso fixed)."""

import os
import time

import numpy as np

from pick_and_place_rl.kinematics import rotation_from_axes, solve_ik
from pick_and_place_rl.robot import APPROACH_LOCAL, ARM_JOINTS, FINGER_LOCAL, ROUTE_JUMP, TiagoScene

TORSO_HEIGHT = 0.35
# A comfortable arm configuration with the hand over the table, used as the IK
# nullspace posture so the redundant arm always resolves to the same "elbow".
Q_SEED = np.array([TORSO_HEIGHT, 1.05, 0.57, -0.83, 1.7, 0.64, -0.89, 0.72])
# Joint waypoints (torso, arm_1..7) of PAL's `home` motion, used in reverse to leave the tuck pose.
Q_WRIST = np.array([TORSO_HEIGHT, 0.5, -1.34, -0.48, 1.94, -1.49, 1.37, -1.58])
Q_UNFOLD = np.array([TORSO_HEIGHT, 0.2, 0.35, -0.2, 1.94, -1.57, 1.37, -1.58])
# Hand height for travelling above the table. Fingertips hang ~0.10 m below the hand frame and the
# object top is at 0.915, so this keeps them ~3 cm clear. PP_HOVER_Z overrides it (experiments only).
HOVER_Z = float(os.environ.get("PP_HOVER_Z", "1.05"))
STEP_DQ = 0.08  # rad, per-step joint change limit on the real arm


class ArmController:
    def __init__(self, scene: TiagoScene, z_offset: float = 0.0):
        """z_offset: real table height minus the 0.815 m the policy was trained with. All hand
        heights seen or commanded here are in the *virtual* frame (real z minus z_offset)."""
        self.z_off = z_offset
        self.sc = scene
        self.chain = scene.chain
        self.r_down = rotation_from_axes(
            APPROACH_LOCAL, FINGER_LOCAL, [0, 0, -1], [0, 1, 0]
        )
        self.free = np.ones(self.chain.n, bool)
        self.free[0] = False  # torso is set once and then held
        self.q_cmd = scene.q().copy()

    # ------------------------------------------------------------------ basics
    def hand_pos(self) -> np.ndarray:
        """Measured grasp-frame position (base frame)."""
        p = self.chain.fk(self.sc.q())[:3, 3].copy()
        p[2] -= self.z_off
        return p

    def cmd_hand_pos(self) -> np.ndarray:
        """Grasp-frame position of the last commanded joint vector."""
        p = self.chain.fk(self.q_cmd)[:3, 3].copy()
        p[2] -= self.z_off
        return p

    def ik(self, target, q0=None, iters=15, pref=Q_SEED):
        q0 = self.q_cmd if q0 is None else q0
        q, ok = solve_ik(
            self.chain,
            q0,
            np.asarray(target, float) + np.array([0.0, 0.0, self.z_off]),
            self.r_down,
            q_pref=pref,
            pref_gain=0.1,
            free=self.free,
            max_iters=iters,
        )
        return q, ok

    def servo(self, target, dt=0.1):
        """One control step: IK toward `target`, command the arm, advance sim time."""
        q_new, _ = self.ik(target)
        if not self.sc.sim:
            # Real arm: the IK's posture preference can ask for a large joint change in one step.
            # Spread it over many steps instead.
            lim = min(STEP_DQ, self.sc.max_joint_speed * dt)  # never command faster than the speed limit
            q_new = self.q_cmd + np.clip(q_new - self.q_cmd, -lim, lim)
        self.q_cmd = q_new
        self.sc.send_arm(self.q_cmd[1:], dt * 1.5)
        self.sc.sleep(dt)

    def move_to(self, target, speed=0.10, dt=0.1, settle=0.3):
        """Straight-line Cartesian move of the commanded hand position."""
        target = np.asarray(target, float)
        cur = self.cmd_hand_pos()
        n = max(1, int(np.linalg.norm(target - cur) / (speed * dt)))
        for k in range(1, n + 1):
            self.servo(cur + (target - cur) * k / n, dt)
        if not self.sc.sim:  # the speed-limited command may trail the waypoint: keep going until it arrives
            for _ in range(300):
                if np.linalg.norm(self.cmd_hand_pos() - target) < 0.005:
                    break
                self.servo(target, dt)
        self.sc.sleep(settle)

    def grip(self, opening, wait=1.2):
        self.sc.send_gripper(opening, wait * 0.7)
        self.sc.sleep(wait)

    # ------------------------------------------------------------ safe routing
    def go_ready(self, over_xy=(0.62, 0.0)):
        """From the tucked arm to hovering above the table, without touching it.

        Follows PAL's own `home` motion in reverse (rotate the wrist, then unfold the
        arm out to the side, where nothing is in the way), then rises high and crosses
        over the table edge in Cartesian space before settling to the hover height.
        """
        sc = self.sc
        sc.send_gripper(0.045, 1.0)
        sc.sleep(0.5 + sc.send_torso(TORSO_HEIGHT, 3.0))
        sc.sleep(0.5 + sc.send_arm(Q_WRIST[1:], 3.0, max_jump=ROUTE_JUMP))
        sc.sleep(0.5 + sc.send_arm(Q_UNFOLD[1:], 5.0, max_jump=ROUTE_JUMP))
        self.hold_still()
        h = self.hand_pos()
        self.move_to([0.26, h[1], 1.15], speed=0.15)  # straight up, still beside the robot
        self.move_to([0.26, -0.15, 1.15], speed=0.15)
        self.move_to([0.50, -0.15, 1.15], speed=0.15)  # across the table edge, high
        self.move_to([over_xy[0], over_xy[1], 1.15], speed=0.15)
        self.reseed([over_xy[0], over_xy[1], HOVER_Z])

    def hold_still(self, timeout=10.0):
        """Wait until the arm has stopped, then make the measured pose the new command basis.

        A route ends when its duration has elapsed, not when the arm has arrived. Taking `q_cmd` from the measured
        joints at that moment left it up to 0.25 rad away from where the arm then settled, and the first servo
        steps swung the hand out across the table."""
        t0 = time.time()
        while time.time() - t0 < timeout and not self.sc.joints_still(ARM_JOINTS, window=0.5):
            pass
        self.q_cmd = self.sc.q().copy()

    def reseed(self, target, tries=30, seed=0):
        """Move (joint space, high above the table) to a clean IK solution for `target`.

        The incremental IK can end up in a poor local minimum (wrist joints at their limits,
        hand tilted). Solving from random restarts and picking the solution closest to
        `Q_SEED` puts the arm back on a well-conditioned branch.
        """
        rng = np.random.default_rng(seed)
        best = None
        for k in range(tries):
            q0 = Q_SEED if k == 0 else np.clip(
                Q_SEED + np.r_[0.0, rng.normal(0, 0.5, 7)], self.chain.lower, self.chain.upper
            )
            q, ok = solve_ik(
                self.chain, q0, np.asarray(target, float) + np.array([0.0, 0.0, self.z_off]), self.r_down,
                q_pref=Q_SEED, pref_gain=0.2, free=self.free, max_iters=300,
            )
            if not ok:
                continue
            cost = np.linalg.norm((q - Q_SEED)[1:])
            if best is None or cost < best[0]:
                best = (cost, q)
        if best is None:
            raise RuntimeError(f"no IK solution for {target}")
        q = best[1]
        self.sc.sleep(0.5 + self.sc.send_arm(q[1:], 4.0, max_jump=ROUTE_JUMP))
        self.hold_still()
