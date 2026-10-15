"""Pick-and-place environment on top of the TIAGo Gazebo scene.

Observation (12): hand xyz | object - hand | pad - hand | finger opening |
                  gripper command (1=close) | object attached (1/0)
Action (4):       hand displacement dx, dy, dz (m, clipped to +-MAX_STEP) | gripper (>0.5 = close)

The hand point is the `gripper_grasping_frame` in the robot base frame (== world frame, the
base is stationary). Ground-truth object / pad poses come from Gazebo (`/gazebo/model_states`);
replacing them with a detector is future work.
"""

import numpy as np

from pick_and_place_rl.arm_control import HOVER_Z, ArmController
from pick_and_place_rl.robot import OBJECT_NAME, PAD_NAME, TiagoScene

GRASP_DZ = 0.06  # hand height above the object's centre when grasping
OBJ_Z = 0.865  # object centre height when resting on the table
PAD_Z = 0.8155  # pad top surface (pad centre) height
MAX_STEP = 0.012  # max hand displacement per control step (m) -> 0.12 m/s
DT = 0.1  # control period (sim seconds)
GRIP_CLOSE_CMD = 0.0245  # finger command: just under the object radius (0.025)
GRIP_OPEN_CMD = 0.045
CLOSED_ON_OBJ = 0.0262  # finger opening at/below which the object is gripped
PLACE_TOL = 0.04  # success: object centre within this xy distance of the pad centre

OBJ_REGION = ((0.58, 0.68), (0.05, 0.28))
PAD_REGION = ((0.58, 0.68), (-0.28, -0.08))
HAND_START_REGION = ((0.55, 0.68), (-0.20, 0.20))
WORKSPACE_LO = np.array([0.45, -0.32, 0.915])
WORKSPACE_HI = np.array([0.72, 0.32, 1.10])

OBS_DIM = 12
ACT_DIM = 4


def expert_action(obs: np.ndarray) -> np.ndarray:
    """Reactive scripted expert: a function of the observation only."""
    h = obs[0:3]
    o = h + obs[3:6]
    p = h + obs[6:9]
    opening, cmd_close, attached = obs[9], obs[10] > 0.5, obs[11] > 0.5

    grasp_z = o[2] + GRASP_DZ
    place_z = p[2] + 0.05 + GRASP_DZ + 0.004
    grip = cmd_close
    target = h.copy()

    if not attached:
        placed = np.linalg.norm(o[:2] - p[:2]) < PLACE_TOL and o[2] < OBJ_Z + 0.03
        if cmd_close:
            if opening > CLOSED_ON_OBJ:
                pass  # fingers still closing (the env attaches the object once gripped)
            else:
                grip = False  # closed on nothing (missed the object): reopen
        elif placed:
            target = np.array([h[0], h[1], HOVER_Z])  # retreat after a successful place
        else:
            xy_err = np.linalg.norm(o[:2] - h[:2])
            if xy_err > 0.012:
                if h[2] < grasp_z + 0.05 and xy_err > 0.02:
                    target = np.array([h[0], h[1], HOVER_Z])  # misaligned and low: rise first
                else:
                    target = np.array([o[0], o[1], HOVER_Z])
            else:
                target = np.array([o[0], o[1], grasp_z])
                if h[2] - grasp_z < 0.006:
                    grip = True
    else:
        xy_err = np.linalg.norm(p[:2] - h[:2])
        if xy_err > 0.012:
            if h[2] < HOVER_Z - 0.04:
                target = np.array([h[0], h[1], HOVER_Z])  # lift first
            else:
                target = np.array([p[0], p[1], HOVER_Z])
        else:
            target = np.array([p[0], p[1], place_z])
            if h[2] - place_z < 0.006:
                grip = False  # release

    delta = np.clip(target - h, -MAX_STEP, MAX_STEP)
    return np.array([*delta, 1.0 if grip else 0.0])


class PickPlaceEnv:
    def __init__(self, scene: TiagoScene, arm: ArmController, max_steps=250, seed=0):
        self.sc = scene
        self.arm = arm
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.grip_cmd = False
        self.attached = False
        self.steps = 0
        self._arm_ready = False
        self.base_resets = 0
        self.sc.detach_object()  # drop any joint left over from a previous run
        self.sc.sleep(0.2)

    # ------------------------------------------------------------------ helpers
    def ensure_base(self, tol=0.01):
        """The kinematics assume the base frame == world frame; re-pin the base if it drifted."""
        b = self.sc.model_pos("tiago")
        if np.linalg.norm(b[:2]) > tol:
            print(f"WARNING: base drifted to {b.round(3)}; re-pinning at the origin", flush=True)
            self.sc.set_model_pos("tiago", [0.0, 0.0, 0.0])
            self.sc.sleep(0.5)
            self.base_resets += 1

    def _uniform(self, region):
        (x0, x1), (y0, y1) = region
        return np.array([self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)])

    def sample_task(self):
        obj_xy = self._uniform(OBJ_REGION)
        pad_xy = self._uniform(PAD_REGION)
        return obj_xy, pad_xy

    def obs(self) -> np.ndarray:
        h = self.arm.hand_pos()
        o = self.sc.model_pos(OBJECT_NAME)
        p = self.sc.model_pos(PAD_NAME)
        return np.array(
            [*h, *(o - h), *(p - h), self.sc.gripper_opening(), float(self.grip_cmd), float(self.attached)]
        )

    # -------------------------------------------------------------------- reset
    def reset(self, obj_xy=None, pad_xy=None, hand_xy=None) -> np.ndarray:
        sc, arm = self.sc, self.arm
        sc.detach_object()  # always (a joint may exist even if we think we are detached)
        self.attached = False
        sc.sleep(0.1)
        self.ensure_base()
        self.grip_cmd = False
        sc.send_gripper(GRIP_OPEN_CMD, 0.6)

        if obj_xy is None or pad_xy is None:
            s_obj, s_pad = self.sample_task()
            obj_xy = s_obj if obj_xy is None else obj_xy
            pad_xy = s_pad if pad_xy is None else pad_xy
        hand_xy = self._uniform(HAND_START_REGION) if hand_xy is None else hand_xy

        if not self._arm_ready:
            arm.go_ready(over_xy=hand_xy)
            self._arm_ready = True
        else:
            h = arm.cmd_hand_pos()
            arm.move_to([h[0], h[1], HOVER_Z], speed=0.2)
            arm.move_to([hand_xy[0], hand_xy[1], HOVER_Z], speed=0.25)
        sc.set_model_pos(PAD_NAME, [pad_xy[0], pad_xy[1], PAD_Z])
        sc.set_model_pos(OBJECT_NAME, [obj_xy[0], obj_xy[1], OBJ_Z + 0.002])
        sc.sleep(0.6)
        self.steps = 0
        return self.obs()

    # --------------------------------------------------------------------- step
    def step(self, action):
        sc, arm = self.sc, self.arm
        action = np.asarray(action, float)
        delta = np.clip(action[:3], -MAX_STEP, MAX_STEP)
        target = np.clip(arm.cmd_hand_pos() + delta, WORKSPACE_LO, WORKSPACE_HI)
        want_close = bool(action[3] > 0.5)
        if want_close != self.grip_cmd:
            self.grip_cmd = want_close
            sc.send_gripper(GRIP_CLOSE_CMD if want_close else GRIP_OPEN_CMD, 1.0)
        arm.servo(target, DT)
        self.ensure_base()

        o = sc.model_pos(OBJECT_NAME)
        h = arm.hand_pos()
        if self.grip_cmd and not self.attached and sc.gripper_opening() <= CLOSED_ON_OBJ:
            if np.linalg.norm((o - h) - np.array([0, 0, -GRASP_DZ])) < 0.03:
                sc.attach_object()
                self.attached = True
        elif (not self.grip_cmd) and self.attached:
            sc.detach_object()
            self.attached = False

        self.steps += 1
        p = sc.model_pos(PAD_NAME)
        on_pad = np.linalg.norm(o[:2] - p[:2]) < PLACE_TOL and o[2] < OBJ_Z + 0.03
        success = on_pad and (not self.attached) and (not self.grip_cmd) and h[2] > HOVER_Z - 0.03
        fell = o[2] < 0.80 or np.linalg.norm(o[:2] - np.array([0.63, 0.0])) > 0.7
        done = success or fell or self.steps >= self.max_steps
        return self.obs(), done, {"success": bool(success), "fell": bool(fell), "steps": self.steps}
