"""Cartesian arm control on top of TiagoScene (top-down grasps, torso fixed)."""

import numpy as np

from pick_and_place_rl.kinematics import rotation_from_axes, solve_ik
from pick_and_place_rl.robot import APPROACH_LOCAL, FINGER_LOCAL, TiagoScene

TORSO_HEIGHT = 0.35
# A comfortable arm configuration with the hand over the table, used as the IK
# nullspace posture so the redundant arm always resolves to the same "elbow".
Q_SEED = np.array([TORSO_HEIGHT, 1.05, 0.57, -0.83, 1.7, 0.64, -0.89, 0.72])
# Hand waypoint (base frame) that is clear of the tucked arm, the robot and the table.
SAFE_HAND = np.array([0.40, -0.05, 1.10])
HOVER_Z = 1.00  # hand height used for travelling above the table


class ArmController:
    def __init__(self, scene: TiagoScene):
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
        return self.chain.fk(self.sc.q())[:3, 3]

    def cmd_hand_pos(self) -> np.ndarray:
        """Grasp-frame position of the last commanded joint vector."""
        return self.chain.fk(self.q_cmd)[:3, 3]

    def ik(self, target, q0=None, iters=15, pref=Q_SEED):
        q0 = self.q_cmd if q0 is None else q0
        q, ok = solve_ik(
            self.chain,
            q0,
            np.asarray(target, float),
            self.r_down,
            q_pref=pref,
            pref_gain=0.1,
            free=self.free,
            max_iters=iters,
        )
        return q, ok

    def servo(self, target, dt=0.1):
        """One control step: IK toward `target`, command the arm, advance sim time."""
        self.q_cmd, _ = self.ik(target)
        self.sc.send_arm(self.q_cmd[1:], dt * 1.5)
        self.sc.sleep(dt)

    def move_to(self, target, speed=0.10, dt=0.1, settle=0.3):
        """Straight-line Cartesian move of the commanded hand position."""
        target = np.asarray(target, float)
        cur = self.cmd_hand_pos()
        n = max(1, int(np.linalg.norm(target - cur) / (speed * dt)))
        for k in range(1, n + 1):
            self.servo(cur + (target - cur) * k / n, dt)
        self.sc.sleep(settle)

    def grip(self, opening, wait=1.2):
        self.sc.send_gripper(opening, wait * 0.7)
        self.sc.sleep(wait)

    # ------------------------------------------------------------ safe routing
    def go_ready(self, over_xy=(0.75, 0.0)):
        """From any arm configuration to hovering above the table, collision-free.

        The tucked arm has its hand under where a table could be, so first lift the
        hand to a point above and in front of the robot (joint-space move), then
        travel horizontally at hover height.
        """
        sc = self.sc
        sc.send_gripper(0.045, 1.0)
        sc.send_torso(TORSO_HEIGHT, 3.0)
        sc.sleep(3.5)
        q_safe, _ = self.ik(SAFE_HAND, q0=Q_SEED, iters=400)
        sc.send_arm(q_safe[1:], 6.0)
        sc.sleep(6.5)
        self.q_cmd = sc.q().copy()
        self.move_to([over_xy[0], over_xy[1], HOVER_Z], speed=0.12)
