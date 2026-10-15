"""Collect (observation, expert action) pairs in the Gazebo pick-and-place env.

Behaviour cloning with DART noise:   --noise 0.008
DAgger (learned policy drives):      --policy weights.npz --beta 0.3
The label is always the expert's action for the state that was actually visited.
"""

import argparse
import os
import time

import numpy as np

from pick_and_place_rl.arm_control import ArmController
from pick_and_place_rl.env import ACT_DIM, OBS_DIM, PickPlaceEnv, expert_action
from pick_and_place_rl.perception import PerceptionNoise
from pick_and_place_rl.policy import MLPPolicy
from pick_and_place_rl.robot import TiagoScene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="/ros_ws/scratch/tiago.urdf")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.0, help="std (m) of noise added to executed xyz actions")
    ap.add_argument("--noisy-fraction", type=float, default=0.7, help="fraction of episodes that use noise")
    ap.add_argument("--policy", default=None, help="npz policy; if set it drives with prob (1-beta)")
    ap.add_argument("--beta", type=float, default=0.5, help="prob. of executing the expert action per step (DAgger)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perception", action="store_true", help="poses from the head-camera pipeline, not Gazebo")
    ap.add_argument("--obs-noise", action="store_true", help="the policy sees camera-like noisy poses; labels stay clean")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    urdf = open(args.urdf).read()
    sc = TiagoScene(urdf, use_perception=args.perception)
    if args.perception:
        from pick_and_place_rl.perception_node import start_inprocess

        start_inprocess(sc, urdf)
    env = PickPlaceEnv(sc, ArmController(sc), seed=args.seed,
                       obs_noise=PerceptionNoise(args.seed) if args.obs_noise else None)
    rng = np.random.default_rng(args.seed + 1000)
    policy = MLPPolicy.load(args.policy) if args.policy else None

    obs_buf, act_buf, ep_buf = [], [], []
    stats = []
    t_start = time.time()
    for ep in range(args.episodes):
        obs = env.reset()
        noisy = args.noise > 0 and rng.random() < args.noisy_fraction
        info = {}
        for _ in range(env.max_steps):
            a_exp = expert_action(env.clean)  # label from the true state
            obs_buf.append(obs.copy())
            act_buf.append(a_exp.copy())
            ep_buf.append(ep)
            if policy is not None and rng.random() > args.beta:
                a_exec = policy.act(obs)
            else:
                a_exec = a_exp.copy()
            if noisy:
                a_exec[:3] += rng.normal(0.0, args.noise, 3)
            obs, done, info = env.step(a_exec)
            if done:
                break
        stats.append((info.get("success", False), info.get("steps", 0)))
        print(
            f"ep {ep + 1}/{args.episodes} success={info.get('success')} fell={info.get('fell')} "
            f"steps={info.get('steps')} noisy={noisy} samples={len(obs_buf)} "
            f"wall={time.time() - t_start:.0f}s",
            flush=True,
        )
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(
            args.out,
            obs=np.array(obs_buf, np.float32).reshape(-1, OBS_DIM),
            act=np.array(act_buf, np.float32).reshape(-1, ACT_DIM),
            episode=np.array(ep_buf, np.int32),
        )
    ok = sum(s for s, _ in stats)
    print(f"DONE: {ok}/{len(stats)} successful episodes, {len(obs_buf)} samples -> {args.out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
