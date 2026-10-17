"""Exercise the interactive commissioning tools (calibrate, grip_test) against Gazebo with a scripted "person".

    python3 -m pick_and_place_rl.selftest calib --ox 0.62 --oy 0.15
    python3 -m pick_and_place_rl.selftest grip

calib: the person nudges the hand until it is centred on the TRUE object (Gazebo ground truth), then the saved
offset must equal true - perceived. grip: runs the candidate loop with "yes" on the first candidate and checks
that it reports a command and does not crash. With sim=False there is no grasp assist, so grip only checks flow.
"""

import argparse
import json
import re
import subprocess
import sys
import time

from pick_and_place_rl.rehearsal import place


def start_perception():
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "pick_and_place_rl.perception_node"],
        stdout=open("/ros_ws/logs/selftest_perception.log", "w"), stderr=subprocess.STDOUT,
    )


def read_until(p, pattern, timeout, log):
    """Read the child's stdout character by character (prompts have no trailing newline) until `pattern` matches."""
    import os
    import select

    t0 = time.time()
    buf = ""
    fd = p.stdout.fileno()
    while time.time() - t0 < timeout:
        r, _, _ = select.select([fd], [], [], 0.5)
        if not r:
            if p.poll() is not None:
                break
            continue
        chunk = os.read(fd, 4096).decode(errors="replace")
        if not chunk:
            break
        buf += chunk
        log.append(chunk)
        print(chunk, end="", flush=True)
        m = re.search(pattern, buf)
        if m:
            return m
    return None


def send(p, text):
    p.stdin.write(text.encode())
    p.stdin.flush()


def calib(args):
    perc = start_perception()
    try:
        place("pick_object", args.ox, args.oy, 0.867)
        place("place_pad", args.px, args.py, 0.8155)
        time.sleep(2.0)
        out = "/ros_ws/data/selftest_calib.json"
        cmd = [sys.executable, "-u", "-m", "pick_and_place_rl.calibrate", "--urdf", "/ros_ws/scratch/tiago.urdf",
               "--speed-scale", str(args.speed_scale), "--out", out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=False, bufsize=0)
        log = []
        send(p, "y\n")
        m = read_until(p, r"perceived object \(virtual frame\): \[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\]", 90, log)
        if not m:
            print("SELFTEST FAIL: calibrate never printed the perceived object")
            return 1
        per = (float(m.group(1)), float(m.group(2)))
        if not read_until(p, r"nudge dx dy", 900, log):
            print("SELFTEST FAIL: calibrate never reached the nudge prompt")
            return 1
        # the person looks, and nudges the hand onto the true object (sim: z-offset free, base is at the origin)
        dx, dy = (args.ox - per[0]) * 100.0, (args.oy - per[1]) * 100.0
        print(f"SELFTEST: perceived ({per[0]:.3f}, {per[1]:.3f}) true ({args.ox:.3f}, {args.oy:.3f}); nudging {dx:.2f} {dy:.2f} cm")
        send(p, f"{dx:.2f} {dy:.2f}\n")
        if not read_until(p, r"nudge dx dy", 120, log):
            print("SELFTEST FAIL: no prompt after the nudge")
            return 1
        send(p, "\n")
        if not read_until(p, r"saved ", 120, log):
            print("SELFTEST FAIL: calibrate did not save")
            return 1
        p.wait(timeout=60)
        c = json.load(open(out))
        off = c["offset"]
        want = (args.ox - per[0], args.oy - per[1])
        err = max(abs(off[0] - want[0]), abs(off[1] - want[1])) * 1000.0
        print(f"SELFTEST: saved offset ({off[0] * 1000:.1f}, {off[1] * 1000:.1f}) mm; expected true-perceived ({want[0] * 1000:.1f}, {want[1] * 1000:.1f}) mm; mismatch {err:.1f} mm")
        print("RESULT calibrate:", "PASS" if err < 5.0 else "FAIL (hand did not reach the commanded position, or frames differ)")
        return 0 if err < 5.0 else 1
    finally:
        perc.terminate()


def grip(args):
    cmd = [sys.executable, "-u", "-m", "pick_and_place_rl.grip_test", "--urdf", "/ros_ws/scratch/tiago.urdf"]
    r = subprocess.run(cmd, input="\ny\n", text=True, capture_output=True, timeout=300)
    print(r.stdout[-3000:])
    print(r.stderr[-2000:])
    ok = r.returncode == 0 and "Use: run_real" in r.stdout and "Traceback" not in r.stderr
    print("RESULT grip_test:", "PASS (flow only: no grasp assist, so holding is not tested)" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["calib", "grip"])
    ap.add_argument("--ox", type=float, default=0.62)
    ap.add_argument("--oy", type=float, default=0.15)
    ap.add_argument("--px", type=float, default=0.62)
    ap.add_argument("--py", type=float, default=-0.20)
    ap.add_argument("--speed-scale", type=float, default=0.3)
    args = ap.parse_args()
    import os

    rc = calib(args) if args.what == "calib" else grip(args)
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
