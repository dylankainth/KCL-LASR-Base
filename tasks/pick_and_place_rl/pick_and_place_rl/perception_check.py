"""Measure perception error in Gazebo against ground truth (and save a sample image).

    python3 -m pick_and_place_rl.perception_check --n 20 --out /ros_ws/data/perception_check.json
"""

import argparse
import json
import threading
import time
import types

import numpy as np
from geometry_msgs.msg import PointStamped
from rclpy.executors import SingleThreadedExecutor

from pick_and_place_rl.env import OBJ_REGION, OBJ_Z, PAD_REGION, PAD_Z
from pick_and_place_rl.head import solve_head
from pick_and_place_rl.perception_node import PerceptionNode
from pick_and_place_rl.robot import OBJECT_NAME, PAD_NAME, TiagoScene

TABLE_CENTRE = (0.63, 0.0, 0.85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="/ros_ws/scratch/tiago.urdf")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    urdf = open(args.urdf).read()
    sc = TiagoScene(urdf)
    pn = PerceptionNode(
        types.SimpleNamespace(
            sim=True, window=3, every=1, camera_frame="",
            rgb="/head_front_camera/rgb/image_raw",
            depth="/head_front_camera/depth/image_raw",
            info="/head_front_camera/rgb/camera_info",
        )
    )
    ex = SingleThreadedExecutor()
    ex.add_node(pn)
    threading.Thread(target=ex.spin, daemon=True).start()

    est = {}

    def cb(name):
        def _cb(msg):
            est[name] = (np.array([msg.point.x, msg.point.y, msg.point.z]), time.time())
        return _cb

    sc.node.create_subscription(PointStamped, "/pick_place/object", cb("object"), 5)
    sc.node.create_subscription(PointStamped, "/pick_place/pad", cb("pad"), 5)

    sc.send_torso(0.35, 2.0)
    sc.sleep(2.5)
    pan, tilt, err, cam = solve_head(urdf, 0.35, TABLE_CENTRE)
    print(f"head pan={pan:.3f} tilt={tilt:.3f} (aim error {np.degrees(err):.1f} deg), camera at {cam.round(3)}", flush=True)
    sc.send_head(pan, tilt, 2.0)
    sc.sleep(3.0)

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.n):
        (ox0, ox1), (oy0, oy1) = OBJ_REGION
        (px0, px1), (py0, py1) = PAD_REGION
        obj = np.array([rng.uniform(ox0, ox1), rng.uniform(oy0, oy1), OBJ_Z + 0.002])
        pad = np.array([rng.uniform(px0, px1), rng.uniform(py0, py1), PAD_Z])
        sc.set_model_pos(OBJECT_NAME, obj)
        sc.set_model_pos(PAD_NAME, pad)
        est.clear()
        sc.sleep(2.0)
        time.sleep(0.5)
        gt_o, gt_p = sc.model_pos(OBJECT_NAME), sc.model_pos(PAD_NAME)
        row = {"i": i, "gt_obj": gt_o.tolist(), "gt_pad": gt_p.tolist()}
        for name, gt in (("object", gt_o), ("pad", gt_p)):
            if name in est:
                e = est[name][0]
                row[name] = e.tolist()
                row[name + "_err_xy_cm"] = float(np.linalg.norm(e[:2] - gt[:2]) * 100)
                row[name + "_err_z_cm"] = float((e[2] - gt[2]) * 100)
            else:
                row[name] = None
        rows.append(row)
        print(
            f"{i}: obj err xy={row.get('object_err_xy_cm', float('nan')):.2f} cm "
            f"z={row.get('object_err_z_cm', float('nan')):.2f} cm | "
            f"pad err xy={row.get('pad_err_xy_cm', float('nan')):.2f} cm "
            f"z={row.get('pad_err_z_cm', float('nan')):.2f} cm",
            flush=True,
        )

    if pn.rgb is not None:
        np.save(args.out.replace(".json", "_rgb.npy"), pn.rgb)

    def stat(key):
        v = np.array([r[key] for r in rows if key in r])
        return {"n": len(v), "mean": float(v.mean()), "max": float(v.max())} if len(v) else None

    summary = {
        "head": {"pan": pan, "tilt": tilt},
        "object_xy_cm": stat("object_err_xy_cm"),
        "object_z_cm": stat("object_err_z_cm"),
        "pad_xy_cm": stat("pad_err_xy_cm"),
        "pad_z_cm": stat("pad_err_z_cm"),
        "missed_object": sum(r["object"] is None for r in rows),
        "missed_pad": sum(r["pad"] is None for r in rows),
        "detection_stats": pn.stats,
        "rows": rows,
    }
    json.dump(summary, open(args.out, "w"), indent=1)
    print("SUMMARY", {k: v for k, v in summary.items() if k != "rows"}, flush=True)
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
