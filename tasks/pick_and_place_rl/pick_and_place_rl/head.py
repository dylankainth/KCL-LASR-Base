"""Aim the head camera at the table (grid search over pan/tilt with the URDF kinematics)."""

import numpy as np

from pick_and_place_rl.kinematics import Chain

CAMERA_FRAME = "head_front_camera_color_optical_frame"
HEAD_JOINTS = ["head_1_joint", "head_2_joint"]


def solve_head(urdf_xml: str, torso: float, target, lo=(-0.6, -1.0), hi=(0.6, 0.7)):
    """Pan/tilt that point the camera's optical axis (+z) at `target` (base frame)."""
    chain = Chain(urdf_xml, "base_footprint", CAMERA_FRAME)
    names = chain.joint_names  # torso_lift_joint, head_1_joint, head_2_joint
    assert names[1:] == HEAD_JOINTS, names
    target = np.asarray(target, float)
    best = None
    for pan in np.linspace(lo[0], hi[0], 49):
        for tilt in np.linspace(lo[1], hi[1], 69):
            T = chain.fk(np.array([torso, pan, tilt]))
            axis = T[:3, 2]
            d = target - T[:3, 3]
            err = np.arccos(np.clip(axis @ d / np.linalg.norm(d), -1, 1))
            if best is None or err < best[0]:
                best = (err, pan, tilt, T[:3, 3])
    return best[1], best[2], best[0], best[3]
