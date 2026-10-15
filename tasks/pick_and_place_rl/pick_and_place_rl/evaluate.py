"""Run a policy (learned or the scripted expert) in the Gazebo env; report success, optionally record video."""

import argparse
import json
import os
import time

import numpy as np

from pick_and_place_rl.arm_control import ArmController
from pick_and_place_rl.env import PickPlaceEnv, expert_action
from pick_and_place_rl.perception import PerceptionNoise
from pick_and_place_rl.policy import MLPPolicy
from pick_and_place_rl.robot import TiagoScene
from pick_and_place_rl.video import VideoRecorder


def arm_hand(env):
    return env.arm.hand_pos()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="/ros_ws/scratch/tiago.urdf")
    ap.add_argument("--policy", default=None, help="npz policy; omit to run the scripted expert")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--video", default=None, help="mp4 path (all episodes in one file)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--perception", action="store_true",
                    help="poses come from the head-camera pipeline (perception_node must run), not Gazebo")
    ap.add_argument("--obs-noise", action="store_true", help="corrupt the poses the policy sees (camera-like)")
    ap.add_argument("--noise-bias", type=float, default=0.010, help="per-episode xy bias std (m) of the noise model")
    ap.add_argument("--noise-jitter", type=float, default=0.004)
    ap.add_argument("--report", default=None, help="json file for per-episode results")
    args = ap.parse_args()

    urdf = open(args.urdf).read()
    sc = TiagoScene(urdf, use_perception=args.perception)
    if args.perception:
        from pick_and_place_rl.perception_node import start_inprocess

        start_inprocess(sc, urdf)
    env = PickPlaceEnv(sc, ArmController(sc), seed=args.seed,
                       obs_noise=PerceptionNoise(args.seed, bias_xy=args.noise_bias, bias_z=0.7 * args.noise_bias,
                                                 jitter=args.noise_jitter) if args.obs_noise else None)
    policy = MLPPolicy.load(args.policy) if args.policy else None
    label = args.label or ("Learned policy (behaviour cloning + DAgger)" if policy else "Scripted expert")
    rec = VideoRecorder(sc, args.video, fps=20) if args.video else None
    results = []
    t0 = time.time()
    for ep in range(args.episodes):
        if rec:
            rec.caption = f"{label} | resetting scene"
        obs = env.reset()
        if rec and ep == 0:
            rec.start()  # skip the one-off arm set-up from the tucked pose
        info = {}
        for t in range(env.max_steps):
            act = policy.act(obs) if policy else expert_action(obs)
            if rec:
                rec.caption = f"{label} | episode {ep + 1}/{args.episodes} | step {t}"
            obs, done, info = env.step(act)
            if done:
                break
        if rec:  # hold the final frame briefly so the outcome is visible
            rec.caption = f"{label} | episode {ep + 1}/{args.episodes} | {'SUCCESS' if info['success'] else 'FAIL'}"
            sc.sleep(1.0)
        o = sc.truth_pos("pick_object")
        p = sc.truth_pos("place_pad")
        err = float(np.linalg.norm(o[:2] - p[:2]))
        results.append({"episode": ep, "success": bool(info["success"]), "steps": int(info["steps"]), "place_error_m": err,
                        "closes": [[round(c, 4) for c in cl] for cl in env.close_log],
                        "final_obj_minus_hand": (o - arm_hand(env)).round(3).tolist(),
                        "final_attached": bool(env.attached)})
        print(f"ep {ep + 1}/{args.episodes} success={info['success']} steps={info['steps']} place_err={err * 100:.1f}cm closes={len(env.close_log)} "
              f"close_err_xy={[round(float(np.hypot(c[0], c[1])) * 100, 1) for c in env.close_log]}cm wall={time.time() - t0:.0f}s", flush=True)
    if rec:
        rec.stop()
    ok = sum(r["success"] for r in results)
    errs = [r["place_error_m"] for r in results if r["success"]]
    print(f"RESULT: {ok}/{len(results)} successful ({100.0 * ok / len(results):.0f}%), mean place error {100 * np.mean(errs) if errs else float('nan'):.1f} cm, base re-pins {env.base_resets}", flush=True)
    if args.report:
        with open(args.report, "w") as f:
            json.dump({"policy": args.policy, "successes": ok, "episodes": len(results), "results": results}, f, indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()
