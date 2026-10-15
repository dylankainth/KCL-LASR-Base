"""Run the pick-and-place policy (or the scripted expert) on the REAL TIAGo.

Prerequisites: preflight passes, perception_node is running, an e-stop is in reach.

    python3 -m pick_and_place_rl.run_real --policy models/policy.npz --episodes 3 \
        --table-height 0.815 --speed-scale 0.3 --step-scale 0.4

Safety: low default speeds, per-command joint speed limit and jump check (robot.py), workspace clamp
(env.py), a confirmation before anything moves, and a prompt before every episode. Ctrl-C holds the
arm where it is (it does not retract). The hardware e-stop is the real safety.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from pick_and_place_rl.arm_control import TORSO_HEIGHT, ArmController
from pick_and_place_rl.env import PickPlaceEnv, expert_action
from pick_and_place_rl.head import solve_head
from pick_and_place_rl.policy import MLPPolicy
from pick_and_place_rl.preflight import fetch_urdf
from pick_and_place_rl.robot import TiagoScene

SIM_TABLE_TOP = 0.815


def ask(prompt, default_no=True):
    try:
        ans = input(f"{prompt} [{'y/N' if default_no else 'Y/n'}] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes") if default_no else ans not in ("n", "no")


def park_step(sc, args, urdf):
    """Survey the table, object and pad; drive the base into the trained region unless --no-park."""
    from pick_and_place_rl.base import BaseDriver
    from pick_and_place_rl.park import auto_park

    base = None if args.no_park else BaseDriver(sc, odom=args.odom)
    if base is None:
        class _Still:  # --no-park: report only, the plan is printed but never executed
            travel_left = 0.0

            def stop(self):
                pass
        base = _Still()
    res = auto_park(sc, base, confirm=lambda t: ask(t), dry_run=args.no_park, need_table=not args.no_park)
    if args.no_park and res.ok and res.msg.startswith("dry run:"):
        from pick_and_place_rl.park import ParkResult

        return ParkResult(False, f"not in the trained region ({res.msg[9:]}); move the robot by hand or drop --no-park", res.iterations, res.history)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=None, help="URDF file (default: robot_description from the robot)")
    ap.add_argument("--policy", default=None, help="policy .npz (omit with --expert)")
    ap.add_argument("--expert", action="store_true", help="run the scripted expert instead of the policy")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--table-height", type=float, default=SIM_TABLE_TOP, help="real table top height in base frame (m)")
    ap.add_argument("--speed-scale", type=float, default=0.3, help="scales the joint speed limit (0.8 rad/s)")
    ap.add_argument("--step-scale", type=float, default=0.4, help="scales the per-step hand displacement")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--report", default="real_run.json")
    ap.add_argument("--auto", action="store_true", help="do not prompt between steps of an episode")
    ap.add_argument("--approach-only", action="store_true",
                    help="never close the gripper: stop above the object at grasp height so a person can check alignment")
    ap.add_argument("--grip-close", type=float, default=None,
                    help="finger command when closing (m per finger); find it with grip_test.py")
    ap.add_argument("--no-park", action="store_true", help="do not drive the base: only report where the objects are relative to the trained region")
    ap.add_argument("--odom", default=None, help="odometry topic for parking (default: /mobile_base_controller/odom or /odom)")
    ap.add_argument("--calib", default=None, help="calibration json from calibrate.py (xy/z offset of the camera estimates)")
    args = ap.parse_args()
    if not args.expert and not args.policy:
        sys.exit("give --policy FILE or --expert")

    import rclpy

    rclpy.init()
    if args.urdf:
        urdf = open(args.urdf).read()
    else:
        tmp = rclpy.create_node("run_real_urdf")
        urdf = fetch_urdf(tmp)
        tmp.destroy_node()
        if not urdf:
            sys.exit("no robot_description from robot_state_publisher; pass --urdf")
    sc = TiagoScene(urdf, node_name="pick_place_real", sim=False, speed_scale=args.speed_scale)
    sc.z_offset = args.table_height - SIM_TABLE_TOP
    if args.calib:
        sc.calib = np.array(json.load(open(args.calib))["offset"], float)
        print(f"camera calibration offset (m): {sc.calib.round(4)}")
    arm = ArmController(sc, z_offset=sc.z_offset)
    env = PickPlaceEnv(sc, arm, max_steps=args.max_steps, step_scale=args.step_scale)
    env.no_close = args.approach_only
    if args.grip_close is not None:
        env.grip_close_cmd = args.grip_close
    policy = None if args.expert else MLPPolicy.load(args.policy)

    print(
        f"\nReal-robot run: {'expert' if args.expert else args.policy}; table {args.table_height:.3f} m; "
        f"speed x{args.speed_scale}, step x{args.step_scale}.\n"
        "The arm will unfold from its tucked pose, rise and move over the table.\n"
        "Clear the area, keep a hand on the e-stop."
    )
    if not ask("Start?"):
        return
    if not ask("Is the arm in its tucked (home) pose, torso down? (the route assumes it)"):
        sys.exit("Move the arm to the tucked pose first (PAL play_motion 'home'), then run again.")

    pan, tilt, err, cam = solve_head(urdf, TORSO_HEIGHT, (0.63, 0.0, args.table_height + 0.03))
    print(f"aiming head: pan {pan:.2f}, tilt {tilt:.2f}")
    sc.send_torso(TORSO_HEIGHT, 8.0)
    sc.send_head(pan, tilt, 4.0)
    sc.sleep(8.5)
    print("Place the object (~5 cm cylinder) and the pad on the table inside the marked region.")
    input("Press Enter when they are placed... ")
    if not sc.wait_perception(timeout=20):
        sys.exit("perception does not see both the object and the pad (is perception_node running?)")
    park_report = None
    try:
        park_report = park_step(sc, args, urdf)
    except Exception as e:
        sys.exit(f"parking failed: {e}")
    if park_report is not None and not park_report.ok:
        sys.exit(f"PARK FAILED: {park_report.msg}. Fix the placement or robot position and run again.")
    o, p = sc.model_pos("pick_object"), sc.model_pos("place_pad")
    print(f"perceived (virtual frame): object {o.round(3)}, pad {p.round(3)}")
    if not ask("Do these look right?"):
        return

    results = []
    try:
        for ep in range(args.episodes):
            if ep > 0:
                input(f"Episode {ep + 1}: reset the object/pad by hand, then press Enter... ")
            obs = env.reset()
            print(f"episode {ep + 1}: object-hand {obs[3:6].round(3)}, pad-hand {obs[6:9].round(3)}")
            if not args.auto and not ask("Run this episode?", default_no=False):
                break
            info = {}
            for t in range(env.max_steps):
                act = expert_action(obs) if args.expert else policy.act(obs)
                obs, done, info = env.step(act)
                if args.approach_only and np.linalg.norm(obs[3:5]) < 0.012 and abs(obs[5] + 0.06) < 0.008:
                    print(f"  hand is above the object at grasp height: object-hand = {obs[3:6].round(3)}")
                    print("  Check by eye that the open fingers straddle the object (it should be centred).")
                    break
                if t % 20 == 0:
                    print(f"  step {t}: hand {obs[0:3].round(3)} grip {obs[10]:.0f} attached {obs[11]:.0f}", flush=True)
                if done:
                    break
            sc.sleep(1.0)
            verdict = ask("Did it succeed (object on the pad)?")
            results.append({"episode": ep, "success_perception": bool(info.get("success")),
                            "success_human": verdict, "steps": int(info.get("steps", 0))})
            print(f"  -> perception says {info.get('success')}, human says {verdict}")
    except KeyboardInterrupt:
        print("\nInterrupted: holding the arm where it is.")
    except Exception:
        import traceback

        traceback.print_exc()
        print("\nERROR: holding the arm where it is.", flush=True)
    finally:
        try:
            sc.send_arm(sc.q()[1:], 0.5)  # hold the current pose
        except Exception:
            pass
        with open(args.report, "w") as f:
            json.dump({"policy": args.policy, "expert": args.expert, "table_height": args.table_height,
                       "results": results}, f, indent=1)
        ok = sum(r["success_human"] for r in results)
        print(f"RESULT: {ok}/{len(results)} succeeded (human-judged); report -> {args.report}")
        os._exit(0)


if __name__ == "__main__":
    main()
