"""Gazebo test of auto-park: displace the base, let it park itself from the camera, check against ground truth.

    python3 -m pick_and_place_rl.park_test --trials 10 [--pnp] [--report park.json]

Each trial: objects at random spots in the trained regions (world frame), the base teleported to a random
offset (+-12 cm, +-10 cm, +-8 deg), auto_park drives it, then ground truth says where it ended up and
whether the object and pad are inside the trained regions of the base frame. With --pnp the scripted expert
then runs the pick-and-place from the parked pose (the base is no longer pinned).
"""

import argparse
import json
import os

import numpy as np

from pick_and_place_rl.arm_control import ArmController
from pick_and_place_rl.base import BaseDriver
from pick_and_place_rl.env import (GRIP_OPEN_CMD, OBJ_REGION, OBJ_Z, PAD_REGION, PAD_Z, PickPlaceEnv, expert_action)
from pick_and_place_rl.park import auto_park
from pick_and_place_rl.perception_node import start_inprocess
from pick_and_place_rl.robot import TiagoScene
from pick_and_place_rl.video import VideoRecorder


def to_base(p_world, base):
    x, y, yaw = base
    d = np.array(p_world[:2]) - np.array([x, y])
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def inside(p, region, margin=0.0):
    (x0, x1), (y0, y1) = region
    return x0 + margin <= p[0] <= x1 - margin and y0 + margin <= p[1] <= y1 - margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="/ros_ws/scratch/tiago.urdf")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--pnp", action="store_true", help="run the scripted expert after parking")
    ap.add_argument("--max-offset", type=float, nargs=3, default=[0.12, 0.10, 8.0], help="x (m), y (m), yaw (deg)")
    ap.add_argument("--odom", default="/ground_truth_odom", help="sim odometry slips badly at low speed; ground truth stands in for a good one")
    ap.add_argument("--report", default=None)
    ap.add_argument("--video", default=None, help="mp4 path: records the displaced start, the park and the pick-and-place")
    args = ap.parse_args()

    urdf = open(args.urdf).read()
    sc = TiagoScene(urdf, use_perception=True)
    start_inprocess(sc, urdf)
    arm = ArmController(sc)
    env = PickPlaceEnv(sc, arm, seed=args.seed)
    env.pin_base = False
    rec = VideoRecorder(sc, args.video, fps=20) if args.video else None
    rng = np.random.default_rng(args.seed)
    mx, my, myaw = args.max_offset
    rows = []
    for k in range(args.trials):
        (ox0, ox1), (oy0, oy1) = OBJ_REGION
        (px0, px1), (py0, py1) = PAD_REGION
        obj = np.array([rng.uniform(ox0, ox1), rng.uniform(oy0, oy1)])
        pad = np.array([rng.uniform(px0, px1), rng.uniform(py0, py1)])
        sc.set_model_pos("place_pad", [pad[0], pad[1], PAD_Z])
        sc.set_model_pos("pick_object", [obj[0], obj[1], OBJ_Z + 0.002])
        start = (float(rng.uniform(-mx, mx)), float(rng.uniform(-my, my)), float(np.radians(rng.uniform(-myaw, myaw))))
        sc.set_model_pos("tiago", [start[0], start[1], 0.0], start[2])
        sc.sleep(1.0)
        base = BaseDriver(sc, odom=args.odom)
        if rec:
            rec.caption = "auto-park from camera"
            rec.start()
        res = auto_park(sc, base, confirm=lambda t: True, log=lambda s: print("   " + s, flush=True))
        sc.sleep(1.5)
        bx = sc.truth_pose2d("tiago")
        o_b, p_b = to_base(sc.truth_pos("pick_object"), bx), to_base(sc.truth_pos("place_pad"), bx)
        in_region = inside(o_b, OBJ_REGION) and inside(p_b, PAD_REGION)
        near_region = inside(o_b, OBJ_REGION, -0.01) and inside(p_b, PAD_REGION, -0.01)  # within 1 cm of the trained region
        row = {"trial": k, "start": [round(v, 3) for v in start], "end": [round(v, 3) for v in bx], "yaw_end_deg": round(float(np.degrees(bx[2])), 2),
               "ok": res.ok, "iterations": res.iterations, "msg": res.msg, "in_region": bool(in_region), "near_region": bool(near_region),
               "obj_base": o_b.round(3).tolist(), "pad_base": p_b.round(3).tolist()}
        if args.pnp and res.ok and in_region:
            sc.detach_object()
            sc.send_gripper(GRIP_OPEN_CMD, 0.6)
            env._arm_ready = False
            if rec:
                rec.caption = "pick and place"
            obs = env._reset_real(None)
            info = {}
            for t in range(env.max_steps):
                obs, done, info = env.step(expert_action(obs))
                if done:
                    break
            if rec:
                rec.stop()
            o_final = sc.truth_pos("pick_object")
            row["pnp_info"] = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in info.items()}
            row["pnp_closes"] = [[round(c, 4) for c in cl] for cl in env.close_log]
            row["pnp_final_obj_world"] = o_final.round(3).tolist()
            row["pnp_final_hand_base"] = arm.hand_pos().round(3).tolist()
            row["pnp_success"] = bool(info.get("success"))
            row["pnp_steps"] = int(info.get("steps", 0))
            if k + 1 < args.trials:
                print("--pnp supports one trial per run (the arm cannot be re-tucked here); stopping", flush=True)
                rows.append(row)
                break
        rows.append(row)
        print(f"trial {k + 1}/{args.trials}: start {row['start']} -> end {row['end']} (yaw {row['yaw_end_deg']} deg) "
              f"park_ok={res.ok} it={res.iterations} in_region={in_region} (within 1 cm: {near_region}) {row.get('pnp_success', '')} {res.msg}", flush=True)
    if rec:
        rec.stop()
    n_ok = sum(r["ok"] and r["in_region"] for r in rows)
    n_near = sum(r["ok"] and r["near_region"] for r in rows)
    print(f"RESULT: parked inside the trained regions in {n_ok}/{len(rows)} trials ({n_near}/{len(rows)} within 1 cm); "
          f"|yaw| mean {np.mean([abs(r['yaw_end_deg']) for r in rows]):.2f} deg" +
          (f"; pick-and-place {sum(r.get('pnp_success', False) for r in rows)}/{sum('pnp_success' in r for r in rows)}" if args.pnp else ""), flush=True)
    if args.report:
        json.dump(rows, open(args.report, "w"), indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()
