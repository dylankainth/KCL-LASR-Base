"""Gazebo tests of the closed-loop mode on the REAL-robot code path (TiagoScene sim=False: wall clock, no teleporting
of the robot, no grasp assist, speed limits, camera poses). Ground truth from Gazebo is used only for scoring, and
faults are injected into the perception topics, so the controllers see what they would see on the robot.

    python3 -m pick_and_place_rl.closed_loop_test SCENARIO [--closed-loop] [--report out.json] ...

Scenarios
  probe_close  finger opening vs time when the gripper closes on the can and on nothing, per close command
  grasp        object-estimate bias grid: where does the grasp land, does the retry search find the can
  stale        camera goes quiet during the approach (hold / resume / abort) and while the hand hides the can
  track        constant joint offset ("sag") injected into the arm commands: does the PI cancel it
  touch        arm stall on contact with the table, and the touchdown release with a pad estimate that is too low
  nominal      one full unperturbed expert episode

Limit (same as rehearsal.py): Gazebo's fingers cannot hold the can, so nothing here says the carried can stays held.
"""

import argparse
import json
import threading
import time
import types

import numpy as np
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from rclpy.executors import SingleThreadedExecutor

from pick_and_place_rl.arm_control import TORSO_HEIGHT, ArmController
from pick_and_place_rl.closed_loop import GraspSearch
from pick_and_place_rl.env import GRASP_DZ, GRIP_OPEN_CMD, OBJ_Z, PAD_Z, PickPlaceEnv, expert_action
from pick_and_place_rl.head import solve_head
from pick_and_place_rl.perception_node import PerceptionNode
from pick_and_place_rl.robot import OBJECT_NAME, PAD_NAME, TiagoScene

OBJ_REST = OBJ_Z + 0.002  # object height when dropped onto the table
PAD_XY = (0.62, -0.20)
OBJ_XY = (0.62, 0.15)
HAND_XY = (0.62, -0.05)  # start far enough from the object (>15 cm) that the hand does not hide it


class GzScene(TiagoScene):
    """The real-robot scene (sim=False) plus Gazebo ground truth and teleporting of models, for scoring only."""

    def __init__(self, urdf, **kw):
        super().__init__(urdf, sim=False, **kw)
        self.node.create_subscription(ModelStates, "/gazebo/model_states", self._on_models, 10)
        self._tp = self.node.create_client(SetEntityState, "/gazebo/set_entity_state")
        t0 = time.time()
        while time.time() - t0 < 20 and (OBJECT_NAME not in self._models or not self._tp.service_is_ready()):
            time.sleep(0.1)

    def teleport(self, name, pos, yaw=0.0):
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose.position.x, req.state.pose.position.y, req.state.pose.position.z = (float(v) for v in pos)
        req.state.pose.orientation.z, req.state.pose.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
        req.state.reference_frame = "world"
        fut = self._tp.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 5:
            time.sleep(0.005)
        return fut.done() and fut.result().success

    def truth(self, name):
        return self.truth_pos(name)

    virtual_gripper = None  # dict(tol_x, tol_y) turns the finger-contact emulation on
    vg_arm = None
    vg_blocked = None  # latched at the moment a close reaches the object's width: was the can between the fingers?

    def gripper_opening(self):
        """With the emulation on, a close that reaches the can's width while the true can is between the fingers
        stalls at its width (0.025); otherwise the fingers go on to the command, as they do on nothing. Gazebo's own
        fingers do neither: they close through the can. Everything else (arm, camera, logic) is real."""
        raw = super().gripper_opening()
        vg = self.virtual_gripper
        if vg is None:
            return raw
        if raw > 0.040:
            self.vg_blocked = None
        elif self.vg_blocked is None and raw < 0.030:
            d = self.truth(OBJECT_NAME) - self.vg_arm.hand_pos()
            self.vg_blocked = bool(abs(d[0]) < vg["tol_x"] and abs(d[1]) < vg["tol_y"] and abs(d[2] + GRASP_DZ) < 0.03)
        return max(raw, 0.025) if self.vg_blocked else raw

    def raw_finger_joints(self):
        with self._lock:
            return [float(self._joints[j]) for j in ("gripper_right_finger_joint", "gripper_left_finger_joint")]


class Faults:
    """Wraps the perception publishers: pause them, bias the object estimate, bias the pad estimate."""

    def __init__(self, pn):
        self.paused = False
        self.obj_bias = np.zeros(3)
        self.pad_bias = np.zeros(3)
        self.sent = {"object": 0, "pad": 0}
        for name in ("object", "pad"):
            pn.pub[name] = self._Gate(self, name, pn.pub[name])

    class _Gate:
        def __init__(self, f, name, inner):
            self.f, self.name, self.inner = f, name, inner

        def publish(self, msg):
            if self.f.paused:
                return
            b = self.f.obj_bias if self.name == "object" else self.f.pad_bias
            msg.point.x += float(b[0])
            msg.point.y += float(b[1])
            msg.point.z += float(b[2])
            self.f.sent[self.name] += 1
            self.inner.publish(msg)


def start(urdf, args):
    sc = GzScene(urdf, speed_scale=args.speed_scale)
    pn = PerceptionNode(types.SimpleNamespace(
        sim=False, window=5, every=2, camera_frame="", table_every=4,
        rgb="/head_front_camera/rgb/image_raw", depth="/head_front_camera/depth/image_raw",
        info="/head_front_camera/rgb/camera_info"))
    ex = SingleThreadedExecutor()
    ex.add_node(pn)
    threading.Thread(target=ex.spin, daemon=True).start()
    faults = Faults(pn)
    sc.send_torso(TORSO_HEIGHT, 4.0)
    pan, tilt, _, _ = solve_head(urdf, TORSO_HEIGHT, (0.63, 0.0, 0.845))
    sc.send_head(pan, tilt, 4.0)
    sc.sleep(5.0)
    arm = ArmController(sc)
    env = PickPlaceEnv(sc, arm, max_steps=args.max_steps)
    env.step_scale = args.step_scale
    env.pin_base = False
    if args.grip_close is not None:
        CLOSE_CMD[True] = CLOSE_CMD[False] = args.grip_close
    return sc, faults, arm, env


def place_scene(sc, faults, obj_xy=OBJ_XY, pad_xy=PAD_XY, obj_bias=(0, 0, 0), pad_bias=(0, 0, 0)):
    sc.teleport(PAD_NAME, [pad_xy[0], pad_xy[1], PAD_Z])
    sc.teleport(OBJECT_NAME, [obj_xy[0], obj_xy[1], OBJ_REST])
    faults.paused = False
    faults.obj_bias, faults.pad_bias = np.array(obj_bias, float), np.array(pad_bias, float)
    sc.sleep(1.5)


CLOSE_CMD = {True: 0.016, False: 0.0245}  # closed loop needs a close that an empty gripper can be told apart from


def fresh_episode(env, closed_loop, hand_xy=HAND_XY, keep_search=False):
    env.closed_loop = closed_loop
    env.grip_close_cmd = CLOSE_CMD[closed_loop]
    if not keep_search:
        env.search = GraspSearch()
    env.attached = False
    env.grip_cmd = False
    env.stale_holds = env.touchdowns = 0
    return env.reset(hand_xy=np.array(hand_xy))


def run_until(env, sc, stop, max_steps, act=expert_action, obs=None, trace=None, every=1):
    """Step the env with `act` until stop(obs, info) or done. Returns (obs, info, steps, error)."""
    info, err, t = {}, None, 0
    obs = env.obs() if obs is None else obs
    try:
        for t in range(max_steps):
            obs, done, info = env.step(act(obs))
            if trace is not None and t % every == 0:
                h = env.arm.hand_pos()
                trace.append({"t": t, "hand": h.round(4).tolist(), "cmd": env.arm.cmd_hand_pos().round(4).tolist(),
                              "corr": env._corr.round(4).tolist(), "open": round(sc.gripper_opening(), 4),
                              "k": env.search.k, "attached": bool(env.attached),
                              "obj_truth": sc.truth(OBJECT_NAME).round(4).tolist()})
            if t % 20 == 0:
                print(f"  step {t}: hand {env.arm.hand_pos().round(3)} perceived obj {sc.model_pos(OBJECT_NAME).round(3)} "
                      f"true obj {sc.truth(OBJECT_NAME).round(3)} open {sc.gripper_opening():.4f} grip {int(env.grip_cmd)} "
                      f"att {int(env.attached)} k {env.search.k}", flush=True)
            if stop(obs, info) or done:
                break
    except RuntimeError as e:
        err = str(e)
    return obs, info, t + 1, err


# ---------------------------------------------------------------------------------------------- scenarios
def debug_steps(sc, faults, arm, env, args):
    """Whole unperturbed episodes with a per-step view of what the expert asks for and what the arm does."""
    out = []
    for rep in range(args.reps):
        place_scene(sc, faults)
        obs = fresh_episode(env, args.closed_loop)
        print(f"--- episode {rep}: start hand {arm.hand_pos().round(3)} cmd {arm.cmd_hand_pos().round(3)} "
              f"q_cmd-q {np.abs(arm.q_cmd - sc.q()).max():.3f} rad", flush=True)
        err, rows = None, []
        for t in range(args.debug_steps):
            a = expert_action(obs)
            try:
                obs, done, info = env.step(a)
            except RuntimeError as e:
                err = str(e)
                break
            rows.append((t, a.round(4).tolist(), arm.cmd_hand_pos().round(3).tolist(), arm.hand_pos().round(3).tolist(),
                         obs[3:6].round(3).tolist(), round(sc.gripper_opening(), 4), int(env.grip_cmd), int(env.attached), env.search.k))
            if t % 10 == 0:
                print("  ", rows[-1], flush=True)
            if done:
                break
        print(f"--- episode {rep} ended after {len(rows)} steps: info {info if rows else None} err {err}", flush=True)
        out.append({"rep": rep, "steps": len(rows), "error": err, "rows": rows[::5]})
    return out


def probe_close(sc, faults, arm, env, args):
    """Close the gripper on the can and on empty table at several commands; log the finger joints over time."""
    out = []
    for cmd in args.closes:
        for target in ("can", "empty"):
            place_scene(sc, faults)
            env.closed_loop = False
            env.reset(hand_xy=np.array(HAND_XY))
            xy = OBJ_XY if target == "can" else (0.62, 0.05)
            if target == "empty":
                sc.teleport(OBJECT_NAME, [0.62, 0.25, OBJ_REST])  # out of the way
                sc.sleep(1.0)
            arm.move_to([xy[0], xy[1], HOVER_DOWN], speed=0.08)
            arm.move_to([xy[0], xy[1], OBJ_REST + GRASP_DZ], speed=0.05)
            sc.sleep(0.8)
            hand = arm.hand_pos()
            can0 = sc.truth(OBJECT_NAME).copy()
            sc.send_gripper(cmd, 1.5)
            ts = []
            t0 = time.time()
            while time.time() - t0 < 2.5:
                ts.append([round(time.time() - t0, 2), *np.round(sc.raw_finger_joints(), 4).tolist()])
                time.sleep(0.1)
            can1 = sc.truth(OBJECT_NAME)
            row = {"command": cmd, "target": target, "hand": hand.round(3).tolist(),
                   "final_fingers": ts[-1][1:], "final_opening": round(float(np.mean(ts[-1][1:])), 4),
                   "can_moved_cm": round(float(np.linalg.norm((can1 - can0)[:2])) * 100, 2),
                   "can_z": round(float(can1[2]), 3), "series": ts[::3]}
            print(f"close {cmd} on {target}: final opening {row['final_opening']} fingers {row['final_fingers']} "
                  f"can moved {row['can_moved_cm']} cm z {row['can_z']}", flush=True)
            out.append(row)
            sc.send_gripper(GRIP_OPEN_CMD, 1.0)
            sc.sleep(1.0)
            arm.move_to([xy[0], xy[1], HOVER_DOWN + 0.12], speed=0.08)
    return out


HOVER_DOWN = 1.0


def grasp(sc, faults, arm, env, args):
    out = []
    biases = [tuple(float(v.replace("m", "-")) for v in b.split(",")) for b in args.biases]
    for rep in range(args.reps):
        for b in biases:
            for closed in ([False, True] if args.both else [args.closed_loop]):
                place_scene(sc, faults, obj_bias=(b[0], b[1], 0.0))
                trace = []
                try:
                    obs = fresh_episode(env, closed)
                except RuntimeError as e:
                    out.append({"bias": b, "closed_loop": closed, "error": str(e)})
                    print("RESET ERROR", e, flush=True)
                    continue
                t0 = time.time()
                obs, info, n, err = run_until(env, sc, lambda o, i: env.attached or env.search_exhausted,
                                              args.max_steps, obs=obs, trace=trace, every=3)
                hand = arm.hand_pos()
                can = sc.truth(OBJECT_NAME)
                rel = (can - hand)[:2]
                row = {"bias_cm": [round(v * 100, 1) for v in b], "closed_loop": closed, "rep": rep,
                       "attached": bool(env.attached), "exhausted": bool(env.search_exhausted), "steps": n,
                       "error": err, "stale_holds": env.stale_holds, "can_minus_hand_cm": (rel * 100).round(2).tolist(),
                       "can_z": round(float(can[2]), 3), "search_k": env.search.k,
                       "learnt_bias_cm": (env.search.bias * 100).round(2).tolist(),
                       "final_opening": round(sc.gripper_opening(), 4), "seconds": round(time.time() - t0, 1)}
                tag = ("GRASPED" if row["attached"] else "NO-GRASP") + (" err" if err else "")
                print(f"bias {row['bias_cm']} cm cl={closed} rep {rep}: {tag} steps {n} k {env.search.k} "
                      f"can-hand {row['can_minus_hand_cm']} cm learnt {row['learnt_bias_cm']} {err or ''}", flush=True)
                row["trace"] = trace[-40:]
                out.append(row)
    return out


def carry(sc, faults, arm, env, args):
    """A long carry with the camera alive: the object estimate is frozen on purpose while it is in the gripper, which
    must not be mistaken for a dead camera. Carried across the table to the pad, then released."""
    out = []
    place_scene(sc, faults)
    obs = fresh_episode(env, True, hand_xy=OBJ_XY)
    env.attached, env.grip_cmd = True, True
    sc.carrying = True
    env.stale_holds = 0
    obs = env.obs()
    t0 = time.time()
    rel, err = None, None
    try:
        for t in range(args.max_steps):
            obs, done, info = env.step(expert_action(obs))
            if not env.attached:
                rel = t
                break
    except RuntimeError as e:
        err = str(e)
    sc.carrying = False
    row = {"released_at_step": rel, "seconds": round(time.time() - t0, 1), "stale_holds": env.stale_holds, "error": err,
           "final_hand": arm.hand_pos().round(3).tolist()}
    print(row, flush=True)
    out.append(row)
    return out


def flow(sc, faults, arm, env, args):
    """Approach, grasp judgement (finger contact emulated), carry, place, release in one closed-loop episode, with a
    small object bias. Ends at the first release. Gazebo's can does not travel with the hand (no grasp assist), so
    'success' is not meaningful; the point is that the stages chain with no hold, abort or stall."""
    sc.virtual_gripper = {"tol_x": args.tol_x, "tol_y": args.tol_y}
    sc.vg_arm = arm
    out = []
    for b in [tuple(float(v.replace("m", "-")) for v in x.split(",")) for x in args.biases]:
        place_scene(sc, faults, obj_bias=(b[0], b[1], 0.0))
        sc.vg_blocked = None
        obs = fresh_episode(env, True)
        state = {"was": False, "t_att": None}

        def stop(o, i):
            if env.attached and not state["was"]:
                state["was"], state["t_att"] = True, env.steps
            return state["was"] and not env.attached

        t0 = time.time()
        obs, info, n, err = run_until(env, sc, stop, args.max_steps, obs=obs)
        row = {"bias_cm": [round(v * 100, 1) for v in b], "gripped_at_step": state["t_att"], "released_at_step": n if state["was"] and not env.attached else None,
               "misses": env.search.misses, "touchdown_releases": env.touchdowns, "stale_holds": env.stale_holds,
               "error": err, "seconds": round(time.time() - t0, 1), "learnt_bias_cm": (env.search.bias * 100).round(2).tolist(),
               "final_hand": arm.hand_pos().round(3).tolist()}
        print(row, flush=True)
        out.append(row)
        sc.carrying = False
    sc.virtual_gripper = None
    return out


def learn(sc, faults, arm, env, args):
    """Object estimate biased by b. Finger contact is EMULATED from ground truth (see GzScene.gripper_opening):
    the can is held if it is within tol_x / tol_y of the hand when the fingers reach its width. Legacy mode (default
    close command, no search) vs closed loop, then a second closed-loop episode with the same bias to see the learnt
    correction carry over."""
    sc.virtual_gripper = {"tol_x": args.tol_x, "tol_y": args.tol_y}
    sc.vg_arm = arm
    out = []
    for b in [tuple(float(v.replace("m", "-")) for v in x.split(",")) for x in args.biases]:
        for closed in (False, True):
            env.search = GraspSearch()
            for ep in range(args.episodes if closed else 1):
                place_scene(sc, faults, obj_bias=(b[0], b[1], 0.0))
                sc.vg_blocked = None
                m0 = env.search.misses
                t0 = time.time()
                try:
                    obs = fresh_episode(env, closed, keep_search=True)
                except RuntimeError as e:
                    out.append({"bias_cm": [round(v * 100, 1) for v in b], "closed_loop": closed, "episode": ep, "error": str(e)})
                    print("RESET ERROR", e, flush=True)
                    continue
                obs, info, n, err = run_until(env, sc, lambda o, i: env.attached or env.search_exhausted, args.max_steps, obs=obs)
                row = {"bias_cm": [round(v * 100, 1) for v in b], "closed_loop": closed, "episode": ep,
                       "believed_gripped": bool(env.attached), "really_gripped": bool(sc.vg_blocked),
                       "misses": env.search.misses - m0, "exhausted": bool(env.search_exhausted), "error": err,
                       "steps": n, "learnt_bias_cm": (env.search.bias * 100).round(2).tolist(),
                       "stale_holds": env.stale_holds, "seconds": round(time.time() - t0, 1),
                       "can_truth": sc.truth(OBJECT_NAME).round(3).tolist()}
                tag = ("BELIEVES GRIPPED" if row["believed_gripped"] else "no grasp") + (" (REALLY gripped)" if row["really_gripped"] else " (NOT between the fingers)")
                print(f"bias {row['bias_cm']} cm closed={closed} ep {ep}: {tag}, misses {row['misses']}, learnt {row['learnt_bias_cm']} {err or ''}", flush=True)
                out.append(row)
    sc.virtual_gripper = None
    return out


def stale(sc, faults, arm, env, args):
    """Camera goes quiet. (1) hand far from the can: hold, then carry on when the camera is back. (2) never comes
    back: abort after STALE_HOLD_STEPS. (3) negative control: the hand is over the can, which is what hides it, so
    the latched estimate is used and nothing is held. (Case 3 as built silences the WHOLE camera, so the pad estimate,
    which the hand does not hide, goes stale too and the hold is correct; the camera-alive occlusion case is covered
    by every learn / carry episode, which report stale_holds == 0.)"""
    out = []
    for case in ("pause_then_resume", "pause_forever", "hand_over_can"):  # hand_over_can = whole camera dead
        place_scene(sc, faults)
        obs = fresh_episode(env, True)
        if case == "hand_over_can":
            # drive until the hand is within 8 cm of the can in xy and below the camera, then silence the camera
            obs, _, n, err = run_until(env, sc, lambda o, i: np.linalg.norm((o[3:5])) < 0.08 and o[2] < 1.0, args.max_steps, obs=obs)
        else:
            obs, _, n, err = run_until(env, sc, lambda o, i: False, 8, obs=obs)
        h0 = arm.hand_pos().copy()
        faults.paused = True
        t0 = time.time()
        trace, aborted, resumed_at = [], None, None
        steps = 150 if case != "hand_over_can" else 60
        for t in range(steps):
            try:
                obs, done, info = env.step(expert_action(obs))
            except RuntimeError as e:
                aborted = {"after_s": round(time.time() - t0, 1), "msg": str(e)}
                break
            trace.append((round(time.time() - t0, 1), env._stale, arm.hand_pos().round(3).tolist()))
            if case == "pause_then_resume" and resumed_at is None and time.time() - t0 > args.pause_for:
                faults.paused = False
                resumed_at = round(time.time() - t0, 1)
            if case == "pause_then_resume" and resumed_at and time.time() - t0 > resumed_at + 4:
                break
        faults.paused = False
        held = [tr for tr in trace if tr[1] > 0]
        moved_in_hold = (float(np.linalg.norm(np.array(held[-1][2]) - np.array(held[0][2]))) if len(held) > 1 else None)
        row = {"case": case, "hand_at_pause": h0.round(3).tolist(), "hold_steps": len(held),
               "max_stale_counter": max([tr[1] for tr in trace] or [0]), "hand_moved_during_hold_cm":
               None if moved_in_hold is None else round(moved_in_hold * 100, 2), "resumed_at_s": resumed_at,
               "aborted": aborted, "final_hand": arm.hand_pos().round(3).tolist(),
               "hand_moved_after_resume_cm": round(float(np.linalg.norm(arm.hand_pos() - np.array(held[-1][2]))) * 100, 2) if held else None}
        print(row, flush=True)
        out.append(row)
        sc.sleep(0.5)
    return out


def track(sc, faults, arm, env, args):
    """Add a constant joint offset to every arm command (an emulated sag) and compare |cmd - hand| with and
    without the PI. The hand is driven along a fixed path with the expert; the metric is the tracking error."""
    out = []
    off = np.zeros(7)
    off[1] = args.sag
    orig = sc.send_arm

    def sag_send(q_arm, *a, **k):
        return orig(np.asarray(q_arm, float) + off, *a, **k)

    for closed in (False, True):
        place_scene(sc, faults)
        sc.send_arm = orig
        obs = fresh_episode(env, closed)
        sc.send_arm = sag_send
        arm.q_cmd = sc.q().copy() - np.r_[0.0, off]  # the command that would give the measured pose
        errs, trace = [], []
        t_end = args.max_steps
        for t in range(t_end):
            obs, done, info = env.step(expert_action(obs))
            if env.attached or done or env.grip_cmd:  # approach only: the tracking error is about moving, not grasping
                break
            if t > 15:
                errs.append(float(np.linalg.norm(arm.cmd_hand_pos() - env._corr - arm.hand_pos())))
            trace.append([t, arm.cmd_hand_pos().round(4).tolist(), arm.hand_pos().round(4).tolist(), env._corr.round(4).tolist()])
        sc.send_arm = orig
        row = {"closed_loop": closed, "sag_rad": args.sag, "steps": len(errs) + 16,
               "median_err_mm": round(float(np.median(errs)) * 1000, 2) if errs else None,
               "p95_err_mm": round(float(np.percentile(errs, 95)) * 1000, 2) if errs else None,
               "max_corr_mm": round(float(np.abs([tr[3] for tr in trace]).max()) * 1000, 2), "trace": trace[::6]}
        print({k: v for k, v in row.items() if k != "trace"}, flush=True)
        out.append(row)
    return out


def virtual_table(sc, arm, z_contact):
    """Emulate a hard surface: clamp every arm command so the hand frame cannot go below z_contact. Returns an undo."""
    orig = sc.send_arm

    def hand_z(qa):
        return float(arm.chain.fk(np.r_[sc.q()[0], qa])[2, 3])

    def clamped(q_arm, *a, **k):
        q_arm = np.asarray(q_arm, float)
        if hand_z(q_arm) < z_contact:
            q0 = sc.q()[1:]
            if hand_z(q0) < z_contact:
                q_arm = q0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(30):
                    m = (lo + hi) / 2
                    lo, hi = (lo, m) if hand_z(q0 + (q_arm - q0) * m) < z_contact else (m, hi)
                q_arm = q0 + (q_arm - q0) * lo
        return orig(q_arm, *a, **k)

    sc.send_arm = clamped
    return lambda: setattr(sc, "send_arm", orig)


def touch_emul(sc, faults, arm, env, args):
    """Carrying the can down onto the pad with a hard surface emulated at the fingertip contact height. The pad
    estimate is biased in z so the hand is blocked at different heights above where the expert wants it."""
    out = []
    z_contact = args.z_contact
    for bias in args.pad_biases:
        place_scene(sc, faults, pad_bias=(0, 0, bias))
        undo = None
        row = {"pad_bias_cm": round(bias * 100, 1)}
        try:
            fresh_episode(env, True, hand_xy=PAD_XY)
            env.attached, env.grip_cmd = True, True
            sc.carrying = True
            env.touchdowns = env.stale_holds = 0
            undo = virtual_table(sc, arm, z_contact)
            place_z = sc.model_pos(PAD_NAME)[2] + 0.05 + GRASP_DZ + 0.004
            row["place_z"] = round(float(place_z), 4)
            row["blocked_above_place_mm"] = round((z_contact - place_z) * 1000, 1)
            obs = env.obs()
            rel = None
            zmin = 9.0
            for t in range(args.max_steps):
                obs, done, info = env.step(expert_action(obs))
                zmin = min(zmin, float(arm.hand_pos()[2]))
                if not env.attached:
                    rel = {"step": t, "hand_z": round(float(arm.hand_pos()[2]), 4), "cmd_z": round(float(arm.cmd_hand_pos()[2]), 4)}
                    break
            row.update({"released": rel, "touchdown_releases": env.touchdowns, "lowest_hand_z": round(zmin, 4)})
        except RuntimeError as e:
            row["error"] = str(e)
        finally:
            if undo:
                undo()
            sc.carrying = False
        print(row, flush=True)
        out.append(row)
        sc.send_gripper(GRIP_OPEN_CMD, 0.6)
        h = arm.cmd_hand_pos()
        arm.move_to([h[0], h[1], 1.05], speed=0.06)
    return out


def touch(sc, faults, arm, env, args):
    out = []
    # A: does the arm stall on the table? Descend over empty table and watch hand z vs command z.
    place_scene(sc, faults)
    sc.teleport(OBJECT_NAME, [0.62, 0.25, OBJ_REST])
    env.closed_loop = False
    env.reset(hand_xy=np.array((0.62, 0.0)))
    arm.move_to([0.62, 0.0, 1.0], speed=0.06)
    rows = []
    for z in np.arange(1.0, 0.86, -0.005):
        arm.move_to([0.62, 0.0, float(z)], speed=0.04, settle=0.25)
        rows.append([round(float(z), 3), round(float(arm.cmd_hand_pos()[2]), 4), round(float(arm.hand_pos()[2]), 4)])
    print("descent (target z, cmd z, measured z):", rows[::2], flush=True)
    out.append({"case": "descent_over_table", "rows": rows})
    arm.move_to([0.62, 0.0, 1.05], speed=0.06)
    return out


SCENARIOS = {"flow": flow, "carry": carry, "learn": learn, "touch_emul": touch_emul, "debug": debug_steps, "probe_close": probe_close, "grasp": grasp, "stale": stale, "track": track, "touch": touch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", help="comma-separated: " + ", ".join(list(SCENARIOS) + ["nominal"]))
    ap.add_argument("--urdf", default="/ros_ws/scratch/tiago.urdf")
    ap.add_argument("--closed-loop", action="store_true")
    ap.add_argument("--both", action="store_true", help="grasp: run every trial with closed loop off and on")
    ap.add_argument("--biases", nargs="+", default=["0,0", "0.02,0", "0,0.02"], help="object estimate bias 'x,y' (m); 'm' prefix = negative, e.g. m0.025,0")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--closes", type=float, nargs="+", default=[0.0245, 0.016])
    ap.add_argument("--grip-close", type=float, default=None)
    ap.add_argument("--speed-scale", type=float, default=0.3)
    ap.add_argument("--step-scale", type=float, default=0.4)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--sag", type=float, default=0.012, help="track: joint offset (rad) added to arm_2")
    ap.add_argument("--tol-x", type=float, default=0.015, help="learn: emulated gripper tolerance along x (m)")
    ap.add_argument("--tol-y", type=float, default=0.020, help="learn: ... along the finger axis y (m)")
    ap.add_argument("--episodes", type=int, default=2, help="learn: closed-loop episodes per bias")
    ap.add_argument("--debug-steps", type=int, default=60)
    ap.add_argument("--pad-biases", type=float, nargs="+", default=[-0.012, -0.005, 0.0, 0.008, 0.012, 0.02],
                    help="touch_emul: pad estimate z error (m)")
    ap.add_argument("--z-contact", type=float, default=0.945, help="touch_emul: hand-frame height of the emulated surface")
    ap.add_argument("--pause-for", type=float, default=6.0)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    urdf = open(args.urdf).read()
    sc, faults, arm, env = start(urdf, args)
    t0 = time.time()
    res = []
    try:
        for name in args.scenario.split(","):
            print(f"=== {name} ===", flush=True)
            if name == "nominal":
                place_scene(sc, faults)
                obs = fresh_episode(env, args.closed_loop)
                obs, info, n, err = run_until(env, sc, lambda o, i: False, args.max_steps, obs=obs)
                part = [{"closed_loop": args.closed_loop, "steps": n, "info": info, "error": err}]
                print(part, flush=True)
            else:
                part = SCENARIOS[name](sc, faults, arm, env, args)
            res.append({"scenario": name, "results": part})
    except Exception:
        import traceback

        traceback.print_exc()
        res.append({"crashed": traceback.format_exc()})
    finally:
        try:
            sc.send_arm(sc.q()[1:], 0.5)
        except Exception:
            pass
    print(f"SCENARIO {args.scenario} DONE in {time.time() - t0:.0f} s", flush=True)
    if args.report:
        with open(args.report, "w") as f:
            json.dump({"scenario": args.scenario, "args": vars(args), "results": res}, f, indent=1, default=str)
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
