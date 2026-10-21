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
from pick_and_place_rl.closed_loop import GraspSearch, HandTracker, touchdown
from pick_and_place_rl.robot import OBJECT_NAME, PAD_NAME, TiagoScene

GRASP_DZ = 0.06  # hand height above the object's centre when grasping
OBJ_Z = 0.865  # object centre height when resting on the table
PAD_Z = 0.8155  # pad top surface (pad centre) height
MAX_STEP = 0.012  # max hand displacement per control step (m) -> 0.12 m/s
DT = 0.1  # control period (sim seconds)
GRIP_CLOSE_CMD = 0.0245  # finger command: just under the object radius (0.025)
GRIP_OPEN_CMD = 0.045
CLOSED_ON_OBJ = 0.0262  # finger opening at/below which the object is gripped
LOST_OPENING = 0.020  # real robot: opening below this while 'holding' means the object slipped out
PLACE_TOL = 0.04  # success: object centre within this xy distance of the pad centre

OBJ_REGION = ((0.58, 0.68), (0.05, 0.28))
PAD_REGION = ((0.58, 0.68), (-0.28, -0.08))
HAND_START_REGION = ((0.55, 0.68), (-0.20, 0.20))
WORKSPACE_LO = np.array([0.45, -0.32, 0.915])
WORKSPACE_HI = np.array([0.72, 0.32, max(1.10, HOVER_Z + 0.05)])

OBS_DIM = 12
ACT_DIM = 4

SCENE_MARGIN = 0.03  # m outside the trained regions that the perceived scene may be before we refuse to move
TOUCH_STEPS = 3  # consecutive contact steps before the placing release fires
STALE_HOLD_STEPS = 100  # control steps the arm waits for a lost camera estimate before giving up
SCENE_Z_TOL = 0.03  # m: perceived object / pad heights must match the table height this closely


def scene_problems(o, p, margin=SCENE_MARGIN, z_tol=SCENE_Z_TOL):
    """Why the perceived object centre `o` / pad centre `p` (base frame, table-height corrected) cannot be trusted.

    Empty list = plausible. Catches a wrong colour blob, a wrong --table-height or a layout outside what the
    policy was trained on, before any arm motion.
    """
    out = []
    for name, q, region, z_ref in (("object", o, OBJ_REGION, OBJ_Z), ("pad", p, PAD_REGION, PAD_Z)):
        (x0, x1), (y0, y1) = region
        if not (x0 - margin <= q[0] <= x1 + margin and y0 - margin <= q[1] <= y1 + margin):
            out.append(f"{name} at x={q[0]:.2f} y={q[1]:.2f} is outside the trained region "
                       f"(x {x0:.2f}..{x1:.2f}, y {y0:.2f}..{y1:.2f}, +-{margin * 100:.0f} cm)")
        if abs(q[2] - z_ref) > z_tol:
            out.append(f"{name} height {q[2]:.3f} m is {abs(q[2] - z_ref) * 100:.0f} cm from the expected {z_ref:.3f} m "
                       "(wrong table height, or the wrong coloured thing was detected)")
    return out


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
    def __init__(self, scene: TiagoScene, arm: ArmController, max_steps=250, seed=0, step_scale=1.0,
                 obs_noise=None):
        self.sc = scene
        self.arm = arm
        self.max_steps = max_steps
        self.clean = None
        self.grip_close_cmd = GRIP_CLOSE_CMD  # finger command when closing (tune on the real gripper)
        self.no_close = False  # approach-only mode: the gripper is never allowed to close
        self.lost_count = 0
        self.search_exhausted = False
        self.close_log = []  # sim only: true (object - hand) error at each close command
        self.step_scale = step_scale  # < 1 slows the hand down (real robot first trials)
        self.obs_noise = obs_noise  # PerceptionNoise or None (sim: corrupts what the policy sees)
        self.rng = np.random.default_rng(seed)
        self.grip_cmd = False
        self.attached = False
        self.steps = 0
        self._arm_ready = False
        self.base_resets = 0
        self._stall = 0  # consecutive steps the commanded target was far from the hand (real robot watchdog)
        self._stall_ref = None
        self.closed_loop = False  # real robot: encoder PI, grasp retry search, touchdown release, stale-camera hold
        self.tracker = HandTracker()
        self.search = GraspSearch()
        self._touch = 0
        self._stale = 0
        self.pin_base = True  # sim: teleport the base back to the origin if it drifts (off when the robot parks itself)
        self.sc.detach_object()  # drop any joint left over from a previous run
        self.sc.sleep(0.2)

    # ------------------------------------------------------------------ helpers
    def ensure_base(self, tol=0.01):
        """The kinematics assume the base frame == world frame; re-pin the base if it drifted."""
        if not self.pin_base:
            return
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

    def object_pos(self) -> np.ndarray:
        """Perceived object position, plus the closed-loop correction while it is not in the gripper."""
        o = self.sc.model_pos(OBJECT_NAME)
        if self.closed_loop and not self.attached:
            o = o + self.search.offset
        return o

    def clean_obs(self) -> np.ndarray:
        h = self.arm.hand_pos()
        o = self.object_pos()
        p = self.sc.model_pos(PAD_NAME)
        return np.array(
            [*h, *(o - h), *(p - h), self.sc.gripper_opening(), float(self.grip_cmd), float(self.attached)]
        )

    def obs(self) -> np.ndarray:
        o = self.clean_obs()
        self.clean = o  # what the expert labels from (the policy only sees the possibly-noisy version)
        return o if self.obs_noise is None else self.obs_noise(o)

    # -------------------------------------------------------------------- reset
    def reset(self, obj_xy=None, pad_xy=None, hand_xy=None) -> np.ndarray:
        sc, arm = self.sc, self.arm
        sc.detach_object()  # always (a joint may exist even if we think we are detached)
        self.attached = False
        self.close_log = []
        sc.sleep(0.1)
        self.ensure_base()
        self.grip_cmd = False
        if self.obs_noise is not None:
            self.obs_noise.new_episode()
        sc.send_gripper(GRIP_OPEN_CMD, 0.6)

        if not sc.sim:
            return self._reset_real(hand_xy)

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
        if sc.use_perception:  # the robot sees the new scene through the camera pipeline
            sc.unlatch_object()
            sc.unlatch_pad()
            sc.sleep(0.5)
            if not sc.wait_perception(timeout=20.0, max_age=2.0):
                raise RuntimeError("perception did not see the object and pad after the reset")
            sc.sleep(1.0)
        self.steps = 0
        return self.obs()

    def _reset_real(self, hand_xy):
        """Real robot: nothing is teleported. A person places the object and pad before this call."""
        sc, arm = self.sc, self.arm
        hand_xy = self._uniform(HAND_START_REGION) if hand_xy is None else hand_xy
        if not self._arm_ready:
            arm.go_ready(over_xy=hand_xy)
            self._arm_ready = True
        else:
            h = arm.cmd_hand_pos()
            arm.move_to([h[0], h[1], HOVER_Z], speed=0.1)
            arm.move_to([hand_xy[0], hand_xy[1], HOVER_Z], speed=0.1)
        sc.unlatch_object()
        sc.unlatch_pad()
        if not sc.wait_perception(timeout=30.0):
            raise RuntimeError("perception never saw both the object and the pad")
        sc.sleep(1.0)
        self.tracker.reset()
        self.search.new_episode()
        self._touch = self._stale = 0
        self.search_exhausted = False
        bad = scene_problems(sc.model_pos(OBJECT_NAME), sc.model_pos(PAD_NAME))
        if bad:
            raise RuntimeError("refusing to move, the perceived scene is not plausible: " + "; ".join(bad))
        self.steps = 0
        return self.obs()

    def _closed_loop_action(self, action):
        """Contact and freshness feedback over the policy's action: hold on a lost camera, release on touchdown."""
        sc, arm = self.sc, self.arm
        action = action.copy()
        if sc.perception_stale(OBJECT_NAME) or sc.perception_stale(PAD_NAME):
            self._stale += 1
            if self._stale > STALE_HOLD_STEPS:
                raise RuntimeError("the camera lost the object or the pad for too long; holding the arm")
            action[:3] = 0.0
            return action
        self._stale = 0
        if self.attached and action[2] < 0.0:
            p = sc.model_pos(PAD_NAME)
            h, cmd = arm.hand_pos(), arm.cmd_hand_pos()
            near = np.linalg.norm(p[:2] - h[:2]) < 0.02 and h[2] - (p[2] + 0.05 + GRASP_DZ) < 0.03
            if near and touchdown(cmd[2], h[2], action[2]):
                self._touch += 1
            else:
                self._touch = 0
            if self._touch >= TOUCH_STEPS:
                print("  touchdown on the pad: releasing", flush=True)
                action[:3] = 0.0
                action[3] = 0.0
        else:
            self._touch = 0
        return action

    def _check_following(self, target, far=0.02, steps=50, still=0.004):
        """Real robot: abort if the hand stays put while the command is far from it (controller rejected or
        aborted the goals, joint limit, collision). Without this the loop would run on silently."""
        h = self.arm.hand_pos()
        if np.linalg.norm(target - h) <= far:
            self._stall, self._stall_ref = 0, None
            return
        if self._stall == 0:
            self._stall_ref = h.copy()
        self._stall += 1
        if self._stall >= steps:
            if np.linalg.norm(h - self._stall_ref) < still:
                rej = getattr(self.sc._arm_pub, "rejected", 0)
                raise RuntimeError(
                    f"arm not following: hand stuck at {h.round(3)} for {steps} steps while the target is {target.round(3)} "
                    f"({rej} arm goals rejected so far). Check the e-stop, controller state, joint limits and obstacles."
                )
            self._stall, self._stall_ref = 0, None

    # --------------------------------------------------------------------- step
    def step(self, action):
        sc, arm = self.sc, self.arm
        action = np.asarray(action, float)
        if self.closed_loop and not sc.sim:
            action = self._closed_loop_action(action)
        delta = np.clip(action[:3], -MAX_STEP, MAX_STEP) * self.step_scale
        target = np.clip(arm.cmd_hand_pos() + delta, WORKSPACE_LO, WORKSPACE_HI)
        if self.closed_loop and not sc.sim:
            hold = bool(action[3] > 0.5) != self.grip_cmd or self._touch > 0
            target = np.clip(target + self.tracker.correction(arm.cmd_hand_pos(), arm.hand_pos(), hold), WORKSPACE_LO, WORKSPACE_HI)
        if not sc.sim:  # real arm: never let the command run more than 4 cm ahead of the measured hand
            lead = target - arm.hand_pos()
            n = np.linalg.norm(lead)
            if n > 0.04:
                target = arm.hand_pos() + lead * (0.04 / n)
        want_close = bool(action[3] > 0.5) and not self.no_close
        if want_close and not self.grip_cmd and sc.sim:  # log how well aligned the grasp really is
            bx, by, byaw = sc.truth_pose2d("tiago")  # the truth is in the world frame, the hand in the base frame
            d = sc.truth_pos(OBJECT_NAME) - np.array([bx, by, 0.0])
            c, s_ = np.cos(-byaw), np.sin(-byaw)
            rel = np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], d[2]]) - arm.hand_pos()
            self.close_log.append([float(rel[0]), float(rel[1]), float(rel[2] + GRASP_DZ)])
        if want_close != self.grip_cmd:
            self.grip_cmd = want_close
            sc.send_gripper(self.grip_close_cmd if want_close else GRIP_OPEN_CMD, 1.0)
        arm.servo(target, DT)
        self.ensure_base()
        if not sc.sim:
            self._check_following(target)

        o = self.object_pos()
        h = arm.hand_pos()
        ot = sc.truth_pos(OBJECT_NAME) if sc.sim else o  # success is judged on ground truth in sim
        if self.grip_cmd and not self.attached and sc.gripper_opening() <= CLOSED_ON_OBJ:
            if np.linalg.norm((o - h) - np.array([0, 0, -GRASP_DZ])) < 0.03:
                sc.attach_object()
                self.attached = True
                if self.closed_loop and not sc.sim:
                    b = self.search.grasped()
                    print(f"  grasp held; standing object correction now {b.round(3)} m", flush=True)
        elif (not self.grip_cmd) and self.attached:
            sc.detach_object()
            self.attached = False
        if (not sc.sim) and self.attached and sc.gripper_opening() < LOST_OPENING:
            # real robot: the fingers closed further than the object is wide, so it slipped out
            print("WARNING: object lost from the gripper; reopening and re-perceiving", flush=True)
            sc.detach_object()
            self.attached = False
            self.grip_cmd = False
            sc.send_gripper(GRIP_OPEN_CMD, 0.6)
            sc.unlatch_object()
            self.lost_count += 1

        if self.closed_loop and not sc.sim and self.grip_cmd and not self.attached and sc.gripper_opening() < LOST_OPENING:
            # the fingers met: nothing was between them. Reopen and try the next point of the search spiral.
            retry = self.search.missed()
            print(f"  grasp closed on nothing; {'retrying at the next search point' if retry else 'search exhausted'}", flush=True)
            self.grip_cmd = False
            sc.send_gripper(GRIP_OPEN_CMD, 0.6)
            self.search_exhausted = not retry

        self.steps += 1
        pt = sc.truth_pos(PAD_NAME) if sc.sim else sc.model_pos(PAD_NAME)
        on_pad = np.linalg.norm(ot[:2] - pt[:2]) < PLACE_TOL and ot[2] < OBJ_Z + 0.03
        success = on_pad and (not self.attached) and (not self.grip_cmd) and h[2] > HOVER_Z - 0.03
        fell = ot[2] < 0.80 or np.linalg.norm(ot[:2] - np.array([0.63, 0.0])) > 0.7
        done = success or fell or self.steps >= self.max_steps or self.search_exhausted
        return self.obs(), done, {"success": bool(success), "fell": bool(fell), "steps": self.steps}
