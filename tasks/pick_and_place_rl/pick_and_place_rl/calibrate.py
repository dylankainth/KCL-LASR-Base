"""Remove the camera's systematic offset by eye (the policy is only as good as the object poses it is given).

The robot hovers above the perceived object, descends to grasp height with the gripper OPEN, and a person
nudges the hand until the open fingers straddle the object symmetrically. The final difference between the
hand and the perceived position is the camera error at that spot; it is saved and added to every estimate.

    python3 -m pick_and_place_rl.calibrate --table-height 0.80 --out calib.json
    # then:  python3 -m pick_and_place_rl.run_real ... --calib calib.json

Do it at two object positions (left and right half of the placement region) and compare; if the offsets differ by
more than ~7 mm the camera extrinsics need a real calibration, not just a constant offset.
"""

import argparse
import json
import os
import sys

import numpy as np

from pick_and_place_rl.arm_control import HOVER_Z, TORSO_HEIGHT, ArmController
from pick_and_place_rl.env import GRASP_DZ
from pick_and_place_rl.head import solve_head
from pick_and_place_rl.preflight import fetch_urdf
from pick_and_place_rl.robot import OBJECT_NAME, TiagoScene

SIM_TABLE_TOP = 0.815


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--table-height", type=float, default=SIM_TABLE_TOP)
    ap.add_argument("--speed-scale", type=float, default=0.3)
    ap.add_argument("--out", default="calib.json")
    args = ap.parse_args()

    import rclpy

    rclpy.init()
    if args.urdf:
        urdf = open(args.urdf).read()
    else:
        tmp = rclpy.create_node("calibrate_urdf")
        urdf = fetch_urdf(tmp)
        tmp.destroy_node()
        if not urdf:
            sys.exit("no robot_description; pass --urdf")
    sc = TiagoScene(urdf, node_name="pick_place_calibrate", sim=False, speed_scale=args.speed_scale)
    sc.z_offset = args.table_height - SIM_TABLE_TOP
    arm = ArmController(sc, z_offset=sc.z_offset)

    print("Place the object on the table in the placement region. The arm will move; keep the e-stop in reach.")
    if input("Start? [y/N] ").strip().lower() not in ("y", "yes"):
        return
    pan, tilt, _, _ = solve_head(urdf, TORSO_HEIGHT, (0.63, 0.0, args.table_height + 0.03))
    sc.send_torso(TORSO_HEIGHT, 8.0)
    sc.send_head(pan, tilt, 4.0)
    sc.sleep(8.5)
    if not sc.wait_perception(timeout=20):
        sys.exit("perception does not see the object and pad (is perception_node running?)")
    o = sc.model_pos(OBJECT_NAME).copy()
    print(f"perceived object (virtual frame): {o.round(3)}")
    try:
        arm.go_ready(over_xy=(o[0], o[1]))
        arm.move_to([o[0], o[1], HOVER_Z], speed=0.08)
        arm.move_to([o[0], o[1], o[2] + GRASP_DZ], speed=0.03)
        print("Hand is at grasp height with the gripper open.")
        print("Look from above/behind: are the two fingers equally spaced from the object?")
        print("x is forward (away from the robot), y is to the robot's LEFT. Enter a nudge in cm, e.g. '0.5 -1'.")
        while True:
            s = input("nudge dx dy (cm), or blank when centred: ").strip()
            if not s:
                break
            try:
                dx, dy = (float(v) / 100.0 for v in s.split())
            except ValueError:
                print("  expected two numbers")
                continue
            arm.move_to(arm.cmd_hand_pos() + np.array([dx, dy, 0.0]), speed=0.02)
        h = arm.hand_pos()
        offset = [float(h[0] - o[0]), float(h[1] - o[1]), 0.0]
        print(f"camera error at this spot: dx {offset[0] * 100:.1f} cm, dy {offset[1] * 100:.1f} cm")
        out = {"offset": offset, "object_perceived": o.tolist(), "table_height": args.table_height}
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"saved {args.out}")
        arm.move_to([h[0], h[1], HOVER_Z], speed=0.05)
    finally:
        try:
            sc.send_arm(sc.q()[1:], 0.5)
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    main()
