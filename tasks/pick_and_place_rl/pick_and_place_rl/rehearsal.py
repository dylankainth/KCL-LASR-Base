"""Rehearse the REAL-robot code path against Gazebo.

Runs `perception_node` and `run_real` (sim=False: wall clock, no teleporting, no grasp assist, speed limits,
camera poses, human prompts answered from a script) against the simulator, the closest stand-in for the real
robot's ROS interface that we have. A person's placing of the objects is done through Gazebo before the run.

Limit: with sim=False the grasp-assist joint is not used, and Gazebo's finger friction cannot hold the can, so
this rehearsal checks everything EXCEPT the grasp itself (arm routes, speed limits, perception, latch, prompts).

    python3 -m pick_and_place_rl.rehearsal --policy models/policy.npz --ox 0.62 --oy 0.15 --px 0.62 --py -0.2
"""

import argparse
import subprocess
import sys
import time


def place(name, x, y, z):
    req = (
        "{state: {name: '%s', pose: {position: {x: %f, y: %f, z: %f}, orientation: {w: 1.0}}, "
        "reference_frame: world}}" % (name, x, y, z)
    )
    subprocess.run(
        ["ros2", "service", "call", "/gazebo/set_entity_state", "gazebo_msgs/srv/SetEntityState", req],
        check=True, capture_output=True, timeout=30,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--expert", action="store_true")
    ap.add_argument("--ox", type=float, default=0.62)
    ap.add_argument("--oy", type=float, default=0.15)
    ap.add_argument("--px", type=float, default=0.62)
    ap.add_argument("--py", type=float, default=-0.20)
    ap.add_argument("--speed-scale", type=float, default=0.3)
    ap.add_argument("--step-scale", type=float, default=0.4)
    args = ap.parse_args()

    perc = subprocess.Popen(
        [sys.executable, "-u", "-m", "pick_and_place_rl.perception_node"],
        stdout=open("/ros_ws/logs/rehearsal_perception.log", "w"), stderr=subprocess.STDOUT,
    )
    try:
        place("pick_object", args.ox, args.oy, 0.867)
        place("place_pad", args.px, args.py, 0.8155)
        time.sleep(2.0)
        cmd = [sys.executable, "-u", "-m", "pick_and_place_rl.run_real", "--episodes", "1", "--auto",
               "--speed-scale", str(args.speed_scale), "--step-scale", str(args.step_scale),
               "--max-steps", "700", "--report", "/ros_ws/data/rehearsal.json",
               "--urdf", "/ros_ws/scratch/tiago.urdf"]
        cmd += ["--expert"] if args.expert else ["--policy", args.policy]
        # answers: Start? y | placed (Enter) | look right? y | succeeded? y
        answers = "y\n\ny\ny\n"
        r = subprocess.run(cmd, input=answers, text=True, capture_output=True, timeout=1500)
        print(r.stdout[-6000:])
        print(r.stderr[-3000:])
        print("EXIT", r.returncode, flush=True)
    finally:
        perc.terminate()
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
