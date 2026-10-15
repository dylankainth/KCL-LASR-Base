"""Find a gripper close command that holds the object, with no arm motion.

    python3 -m pick_and_place_rl.grip_test [--urdf FILE]

For each candidate command (finger position per finger, m; object radius is 0.025 for the trained size) the
gripper opens, you place the object between the fingers, it closes, and you tug the object gently by hand.
Use the largest (gentlest) command that holds; pass it to run_real as --grip-close.
"""

import argparse
import os
import sys

from pick_and_place_rl.preflight import fetch_urdf
from pick_and_place_rl.robot import TiagoScene

CANDIDATES = [0.0245, 0.0225, 0.0205, 0.0185, 0.0165]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=None)
    args = ap.parse_args()
    import rclpy

    rclpy.init()
    if args.urdf:
        urdf = open(args.urdf).read()
    else:
        tmp = rclpy.create_node("grip_test_urdf")
        urdf = fetch_urdf(tmp)
        tmp.destroy_node()
        if not urdf:
            sys.exit("no robot_description; pass --urdf")
    sc = TiagoScene(urdf, node_name="pick_place_grip_test", sim=False)
    held_with = None
    try:
        for cmd in CANDIDATES:
            sc.send_gripper(0.045, 1.5)
            sc.sleep(2.0)
            input(f"[{cmd * 1000:.1f} mm] hold the object between the open fingers, then press Enter... ")
            sc.send_gripper(cmd, 1.5)
            sc.sleep(2.5)
            print(f"  finger opening now {sc.gripper_opening() * 1000:.1f} mm (object radius ~25 mm)")
            ans = input("  let go of the object. Does it stay in the gripper, and resist a gentle tug? [y/N] ")
            if ans.strip().lower() in ("y", "yes"):
                held_with = cmd
                print(f"  -> holds with {cmd}")
                break
            print("  -> slips; squeezing harder")
        sc.send_gripper(0.045, 1.0)
        sc.sleep(1.5)
        if held_with is None:
            print("No candidate held the object: change the object/finger pads or raise the gripper effort limit.")
        else:
            print(f"Use: run_real ... --grip-close {held_with}")
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
