"""Stress-test perception.py / table.py on a synthetic, deliberately imperfect camera (numpy only, no ROS).

    python3 -m pick_and_place_rl.synthetic_check [--n 40] [--seed 1] [--only NAME] [--report out.json]

A ray-cast pinhole camera (head pose like the operating pose: ~1.25 m high, looking at the table centre) renders
the table, a blue 5x10 cm can and a red pad as RGB + depth, then degrades them the way a real structured-light /
ToF head camera does. The NOISE numbers are assumptions (typical Xtion/Astra-class figures), NOT measurements of
the real TIAGo camera; the point is to see which imperfection breaks the pipeline first and by how much.

Per scenario it reports the object / pad error against ground truth (base frame, cm), the detection rate, and the
table-fit error (height, edge distance, yaw). "bad" = object error > 1 cm (the grasp slack is ~1.5 cm per side).
"""

import argparse
import json
from dataclasses import dataclass

import numpy as np

from pick_and_place_rl import perception as P
from pick_and_place_rl.table import fit_table, points_from_depth

W, H_IMG = 640, 480
K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
CAM_POS = np.array([0.12, 0.0, 1.25])  # base_footprint frame
LOOK_AT = np.array([0.63, 0.0, 0.85])
TABLE_Z = 0.815  # m
EDGE_X = 0.52  # near table edge, m ahead of base_footprint
CAN_R, CAN_H = 0.025, 0.10
PAD_HALF = 0.07


def lookat(pos, target):
    """4x4 transform base <- optical frame (z forward, x right, y down)."""
    f = target - pos
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    T = np.eye(4)
    T[:3, :3] = np.stack([r, d, f], 1)
    T[:3, 3] = pos
    return T


def tilt(T, deg):
    """Rotate the camera about its own x axis (a head-tilt error the TF does not know about)."""
    a = np.radians(deg)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    out = T.copy()
    out[:3, :3] = T[:3, :3] @ R
    return out


@dataclass
class Scenario:
    name: str
    brightness: float = 1.0
    gamma: float = 1.0
    cast: tuple = (1.0, 1.0, 1.0)  # per-channel gain (white balance / coloured light)
    pixel_noise: float = 2.0  # rgb grey levels
    depth_a: float = 0.0010  # sigma(z) = a + b z^2 (m)
    depth_b: float = 0.0040
    depth_mm: bool = True  # publish as 16UC1 millimetres (else float metres)
    dropout: float = 0.01  # random invalid depth
    flying: bool = True  # blended depth at depth discontinuities
    reg_px: tuple = (0, 0)  # depth shifted relative to rgb (registration error), pixels (dx, dy)
    tilt_deg: float = 0.0  # unmodelled camera tilt
    specular: bool = False  # bright glare on the can
    distractor: bool = False  # a blue and a red thing elsewhere in view
    table_rgb: tuple = (196, 160, 118)
    can_rgb: tuple = (28, 60, 200)
    pad_rgb: tuple = (205, 30, 35)
    notes: str = ""


SCENARIOS = [
    Scenario("clean"),
    Scenario("real depth noise", notes="baseline: sigma 1+4z^2 mm, 1 mm quantisation, 1 % dropout, flying pixels"),
    Scenario("noisier depth x2", depth_a=0.002, depth_b=0.008, dropout=0.03),
    Scenario("dim room", brightness=0.45, pixel_noise=4.0),
    Scenario("very dim room", brightness=0.28, pixel_noise=6.0),
    Scenario("bright / washed out", brightness=1.5, gamma=0.8),
    Scenario("warm lights", cast=(1.15, 1.0, 0.72)),
    Scenario("cool lights", cast=(0.82, 1.0, 1.2)),
    Scenario("glare on can", specular=True),
    Scenario("distractors", distractor=True),
    Scenario("depth shifted 2 px", reg_px=(2, 1)),
    Scenario("depth shifted 4 px", reg_px=(4, 2)),
    Scenario("head tilt 0.5 deg", tilt_deg=0.5, notes="systematic: removed by calibrate.py"),
    Scenario("head tilt 1.0 deg", tilt_deg=1.0, notes="systematic: removed by calibrate.py"),
    Scenario("pale blue can", can_rgb=(70, 110, 190)),
    Scenario("dark red pad", pad_rgb=(140, 40, 45)),
    Scenario("white table", table_rgb=(235, 235, 230), brightness=1.3),
    Scenario("blue-ish table", table_rgb=(120, 140, 170)),
    Scenario("float metres depth", depth_mm=False),
]


def render(sc: Scenario, obj_xy, pad_xy, rng, T=None):
    """-> rgb uint8 (H, W, 3), depth (H, W) float metres (before encoding), ground-truth T used."""
    T = lookat(CAM_POS, LOOK_AT) if T is None else T
    u, v = np.meshgrid(np.arange(W), np.arange(H_IMG))
    dirs_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    d = dirs_c @ T[:3, :3].T
    o = T[:3, 3]
    inf = np.inf
    t_best = np.full((H_IMG, W), inf)
    col = np.zeros((H_IMG, W, 3), float)

    def plane(z, mask_fn, rgb, shade=1.0):
        nonlocal t_best, col
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - o[2]) / d[..., 2]
        p = o + d * t[..., None]
        ok = (t > 0.05) & mask_fn(p) & (t < t_best)
        t_best = np.where(ok, t, t_best)
        col[ok] = np.array(rgb, float) * shade

    plane(0.0, lambda p: np.ones(p.shape[:2], bool), (95, 95, 100), 0.8)  # floor
    plane(TABLE_Z, lambda p: (p[..., 0] > EDGE_X) & (p[..., 0] < 1.4) & (np.abs(p[..., 1]) < 0.7), sc.table_rgb)
    px, py = pad_xy
    plane(
        TABLE_Z + 0.002,
        lambda p: (np.abs(p[..., 0] - px) < PAD_HALF) & (np.abs(p[..., 1] - py) < PAD_HALF),
        sc.pad_rgb,
    )
    if sc.distractor:  # a blue folder (flat) and a red mug-ish square, both on the table off to the sides
        plane(
            TABLE_Z + 0.004,
            lambda p: (np.abs(p[..., 0] - 0.95) < 0.10) & (np.abs(p[..., 1] - 0.45) < 0.08),
            (35, 70, 190),
        )
        plane(
            TABLE_Z + 0.03,
            lambda p: (np.abs(p[..., 0] - 0.95) < 0.04) & (np.abs(p[..., 1] + 0.45) < 0.04),
            (190, 35, 40),
        )
    # can: vertical cylinder (side + top cap)
    cx, cy = obj_xy
    ox, oy = o[0] - cx, o[1] - cy
    a = d[..., 0] ** 2 + d[..., 1] ** 2
    b = 2 * (ox * d[..., 0] + oy * d[..., 1])
    c = ox**2 + oy**2 - CAN_R**2
    disc = b**2 - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        t_side = (-b - np.sqrt(np.where(disc >= 0, disc, np.nan))) / (2 * a)
    zs = o[2] + d[..., 2] * t_side
    ok = np.isfinite(t_side) & (t_side > 0.05) & (zs > TABLE_Z) & (zs < TABLE_Z + CAN_H) & (t_side < t_best)
    nx = (o[0] + d[..., 0] * t_side - cx) / CAN_R
    shade = 0.55 + 0.45 * np.clip(-nx, -1, 1) * 0.5 + 0.2  # lit from the robot side
    t_best = np.where(ok, t_side, t_best)
    col[ok] = (np.array(sc.can_rgb, float) * shade[..., None])[ok]
    top_z = TABLE_Z + CAN_H
    plane(top_z, lambda p: (p[..., 0] - cx) ** 2 + (p[..., 1] - cy) ** 2 < CAN_R**2, sc.can_rgb, 1.0)
    # the ray direction has optical z == 1, so the ray parameter t is the optical-frame depth
    depth = np.where(np.isfinite(t_best), t_best, 0.0)

    # ---- appearance
    col = col * np.array(sc.cast)
    if sc.specular:
        # glare: whitish patch on the can's lit side, found from the can mask
        can_mask = np.all(np.abs(col - np.array(sc.can_rgb) * np.array(sc.cast) * 0.9) < 120, -1) & (t_best < inf)
        ys, xs = np.nonzero((np.abs(t_best - t_best[can_mask].min()) < 0.2) & can_mask) if can_mask.any() else ([], [])
        if len(xs):
            gx, gy = xs.mean() - 3, ys.mean()
            glare = np.exp(-(((np.arange(W)[None, :] - gx) / 6.0) ** 2 + ((np.arange(H_IMG)[:, None] - gy) / 12.0) ** 2))
            col = col + glare[..., None] * 230 * (can_mask[..., None])
    col = 255 * np.clip(col * sc.brightness / 255.0, 0, 1) ** sc.gamma
    col = col + rng.normal(0, sc.pixel_noise, col.shape)
    rgb = np.clip(col, 0, 255).astype(np.uint8)
    return rgb, depth, T


def degrade_depth(depth, sc: Scenario, rng):
    d = depth.copy()
    valid = d > 0
    sig = sc.depth_a + sc.depth_b * d**2
    d = np.where(valid, d + rng.normal(0, 1, d.shape) * sig, 0.0)
    if sc.flying:
        # blend at depth discontinuities: a pixel next to a much nearer/farther one reads in between (or nothing)
        gy = np.maximum(np.abs(np.roll(depth, 1, 0) - depth), np.abs(np.roll(depth, -1, 0) - depth))
        gx = np.maximum(np.abs(np.roll(depth, 1, 1) - depth), np.abs(np.roll(depth, -1, 1) - depth))
        edge = (np.maximum(gx, gy) > 0.03) & valid
        r = rng.random(d.shape)
        mix = rng.random(d.shape)
        other = np.where(np.abs(np.roll(depth, 1, 1) - depth) >= np.abs(np.roll(depth, -1, 1) - depth), np.roll(depth, 1, 1), np.roll(depth, -1, 1))
        d = np.where(edge & (r < 0.5), d + mix * (other - d), d)
        d = np.where(edge & (r > 0.8), 0.0, d)
    d = np.where(rng.random(d.shape) < sc.dropout, 0.0, d)
    dx, dy = sc.reg_px
    if dx or dy:
        d = np.roll(np.roll(d, dy, 0), dx, 1)
    d = np.clip(d, 0, 10)
    if sc.depth_mm:
        return np.round(d * 1000).astype(np.uint16), "16UC1"
    return d.astype(np.float32), "32FC1"


def run_scenario(sc: Scenario, n, seed):
    rng = np.random.default_rng(seed)
    obj_err, pad_err, obj_z, rows = [], [], [], []
    miss_o = miss_p = 0
    tab_h, tab_e, tab_y, tab_miss = [], [], [], 0
    for i in range(n):
        obj = np.array([rng.uniform(0.60, 0.66), rng.uniform(0.07, 0.26)])
        pad = np.array([rng.uniform(0.60, 0.66), rng.uniform(-0.26, -0.10)])
        T_true = lookat(CAM_POS, LOOK_AT)
        rgb, depth, _ = render(sc, obj, pad, rng, tilt(T_true, sc.tilt_deg) if sc.tilt_deg else T_true)
        dm, enc = degrade_depth(depth, sc, rng)
        dm = P.depth_to_metres(dm, enc)
        o_est, p_est = P.detect(rgb, dm, K, T_true)  # the pipeline only knows the nominal TF
        if o_est is None:
            miss_o += 1
        else:
            obj_err.append(np.linalg.norm(o_est[:2] - obj) * 100)
            obj_z.append((o_est[2] - (TABLE_Z + CAN_H / 2)) * 100)
        if p_est is None:
            miss_p += 1
        else:
            pad_err.append(np.linalg.norm(p_est[:2] - pad) * 100)
        fit, why = fit_table(points_from_depth(dm, K, T_true))
        if fit is None:
            tab_miss += 1
        else:
            tab_h.append((fit.height - TABLE_Z) * 100)
            tab_e.append((fit.edge_x - EDGE_X) * 100)
            tab_y.append(np.degrees(fit.yaw_err))
        rows.append(why if fit is None else "ok")

    def s(v):
        v = np.array(v)
        return None if not len(v) else {"mean": round(float(v.mean()), 2), "p95": round(float(np.percentile(np.abs(v), 95)), 2), "max": round(float(np.abs(v).max()), 2)}

    oe = np.array(obj_err)
    return {
        "scenario": sc.name,
        "notes": sc.notes,
        "n": n,
        "object_missed": miss_o,
        "pad_missed": miss_p,
        "object_xy_cm": s(obj_err),
        "object_z_cm": s(obj_z),
        "pad_xy_cm": s(pad_err),
        "bad_over_1cm": int((oe > 1.0).sum()) + miss_o,
        "table_missed": tab_miss,
        "table_height_cm": s(tab_h),
        "table_edge_cm": s(tab_e),
        "table_yaw_deg": s(tab_y),
        "table_reasons": sorted(set(r for r in rows if r != "ok")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default=None, help="substring of a scenario name")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    out = []
    print(f"{'scenario':24s} {'obj miss':>8s} {'obj xy mean/p95/max cm':>26s} {'z mean cm':>9s} {'pad xy mean/p95':>16s} {'bad>1cm':>7s} | {'tbl miss':>8s} {'h cm':>6s} {'edge cm':>8s} {'yaw deg':>8s}")
    for sc in SCENARIOS:
        if args.only and args.only not in sc.name:
            continue
        r = run_scenario(sc, args.n, args.seed)
        out.append(r)
        f = lambda d, k: "   -" if d is None else f"{d[k]:.2f}"
        o, p, z = r["object_xy_cm"], r["pad_xy_cm"], r["object_z_cm"]
        print(
            f"{r['scenario']:24s} {r['object_missed']:8d} {f(o, 'mean'):>8s}/{f(o, 'p95'):>6s}/{f(o, 'max'):>6s}   {f(z, 'mean'):>9s} "
            f"{f(p, 'mean'):>7s}/{f(p, 'p95'):>6s} {r['bad_over_1cm']:7d} | {r['table_missed']:8d} {f(r['table_height_cm'], 'p95'):>6s} "
            f"{f(r['table_edge_cm'], 'p95'):>8s} {f(r['table_yaw_deg'], 'p95'):>8s}",
            flush=True,
        )
        if r["table_reasons"]:
            print("      table:", "; ".join(r["table_reasons"]))
    if args.report:
        json.dump(out, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()
