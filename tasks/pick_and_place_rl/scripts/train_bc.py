#!/usr/bin/env python3
"""Train the pick-and-place policy by behaviour cloning (+ DART / DAgger data).

Runs on the host with PyTorch (no ROS needed); exports an .npz for numpy inference
inside the container (`pick_and_place_rl.policy.MLPPolicy`).

    python train_bc.py --data a.npz b.npz --out policy.npz
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

MAX_STEP = 0.012  # keep in sync with pick_and_place_rl.env.MAX_STEP


def load(paths):
    obs, act, ep = [], [], []
    offset = 0
    for p in paths:
        d = np.load(p)
        obs.append(d["obs"])
        act.append(d["act"])
        ep.append(d["episode"] + offset)
        offset += int(d["episode"].max()) + 1
    return np.concatenate(obs), np.concatenate(act), np.concatenate(ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    obs, act, ep = load(args.data)
    n_ep = int(ep.max()) + 1
    val_eps = set(rng.choice(n_ep, max(1, n_ep // 10), replace=False).tolist())
    is_val = np.array([e in val_eps for e in ep])
    print(f"{len(obs)} samples from {n_ep} episodes ({is_val.sum()} validation samples)")

    mean = obs.mean(0)
    std = np.maximum(obs.std(0), 1e-3)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor((obs - mean) / std, dtype=torch.float32, device=dev)
    y_delta = torch.tensor(np.clip(act[:, :3] / MAX_STEP, -1, 1), dtype=torch.float32, device=dev)
    y_grip = torch.tensor(act[:, 3], dtype=torch.float32, device=dev)
    val = torch.tensor(is_val, device=dev)
    tr = ~val

    layers, d_in = [], x.shape[1]
    for _ in range(args.layers):
        layers += [nn.Linear(d_in, args.hidden), nn.ReLU()]
        d_in = args.hidden
    layers += [nn.Linear(d_in, 4)]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bce = nn.BCEWithLogitsLoss()

    def losses(mask):
        out = net(x[mask])
        l_delta = ((torch.tanh(out[:, :3]) - y_delta[mask]) ** 2).mean()
        l_grip = bce(out[:, 3], y_grip[mask])
        acc = ((out[:, 3] > 0).float() == y_grip[mask]).float().mean()
        return l_delta, l_grip, acc

    idx = torch.nonzero(tr).squeeze(1)
    best = (1e9, None)
    for epoch in range(args.epochs):
        net.train()
        perm = idx[torch.randperm(len(idx), device=dev)]
        for i in range(0, len(perm), args.batch):
            b = perm[i : i + args.batch]
            out = net(x[b])
            loss = ((torch.tanh(out[:, :3]) - y_delta[b]) ** 2).mean() + 0.5 * bce(out[:, 3], y_grip[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                ld, lg, acc = losses(tr)
                vd, vg, vacc = losses(val)
            print(f"epoch {epoch:4d} train delta {ld:.4f} grip {lg:.4f} acc {acc:.3f} | val delta {vd:.4f} grip {vg:.4f} acc {vacc:.3f}")
            if vd + vg < best[0]:
                best = (float((vd + vg).cpu()), [(m.weight.detach().cpu().numpy().T.copy(), m.bias.detach().cpu().numpy().copy()) for m in net if isinstance(m, nn.Linear)])

    # export the final network (validation loss is only used for reporting)
    weights = {"obs_mean": mean.astype(np.float32), "obs_std": std.astype(np.float32)}
    for i, m in enumerate(m for m in net if isinstance(m, nn.Linear)):
        weights[f"w{i}"] = m.weight.detach().cpu().numpy().T.astype(np.float32).copy()
        weights[f"b{i}"] = m.bias.detach().cpu().numpy().astype(np.float32).copy()
    np.savez(args.out, **weights)
    print("saved", args.out)


if __name__ == "__main__":
    main()
