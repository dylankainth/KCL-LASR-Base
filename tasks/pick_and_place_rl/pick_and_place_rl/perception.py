"""Colour + depth perception of the pick object and the place pad (base frame).

Pipeline (the same code runs on the Gazebo head camera and on the real TIAGo camera):
  RGB  -> colour mask (HSV thresholds)  -> pixels of the object / pad
  depth -> deproject those pixels with the camera intrinsics -> 3D points
  TF   -> optical frame -> base_footprint
  Object: centre of the top cap (the highest points) minus half the object height.
  Pad:    centroid of the flat patch.
The numpy functions are ROS free so they can be unit tested and reused offline.
"""

from dataclasses import dataclass

import numpy as np

OBJECT_HEIGHT = 0.10  # m, used to turn the top-cap height into the centre height
PAD_THICKNESS = 0.002


@dataclass
class ColourRange:
    """HSV window. Hue in degrees [0, 360); wraps if lo > hi."""

    hue_lo: float
    hue_hi: float
    sat_min: float
    val_min: float

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        h, s, v = rgb_to_hsv(rgb)
        if self.hue_lo <= self.hue_hi:
            hue_ok = (h >= self.hue_lo) & (h <= self.hue_hi)
        else:
            hue_ok = (h >= self.hue_lo) | (h <= self.hue_hi)
        return hue_ok & (s >= self.sat_min) & (v >= self.val_min)


# Blue object, red pad (the sim scene). Change these for the real colours, see the runbook.
BLUE = ColourRange(190.0, 260.0, 0.45, 0.20)
RED = ColourRange(345.0, 15.0, 0.55, 0.25)


def rgb_to_hsv(rgb: np.ndarray):
    """uint8 or float RGB (H, W, 3) -> hue (deg), saturation, value, each (H, W)."""
    x = rgb.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = (60.0 * ((g - b)[rm] / d[rm])) % 360.0
    h[gm] = 60.0 * ((b - r)[gm] / d[gm]) + 120.0
    h[bm] = 60.0 * ((r - g)[bm] / d[bm]) + 240.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return h, s, mx


def clean_mask(mask: np.ndarray, min_pixels: int = 30) -> np.ndarray:
    """Keep the largest 4-connected blob (iterative flood fill on a small grid, no scipy/cv2)."""
    if mask.sum() < min_pixels:
        return np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    seen = np.zeros_like(mask, bool)
    best = None
    h, w = mask.shape
    for y0, x0 in zip(ys[::7], xs[::7]):  # a coarse seed set is enough to find every big blob
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if best is None or len(pts) > len(best):
            best = pts
    if best is None or len(best) < min_pixels:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    arr = np.array(best)
    out[arr[:, 0], arr[:, 1]] = True
    return out


def depth_to_metres(depth: np.ndarray, encoding: str) -> np.ndarray:
    """Gazebo publishes 32FC1 (m); most real drivers publish 16UC1 (mm)."""
    if encoding == "16UC1":
        d = depth.astype(np.float32) * 1e-3
    else:
        d = depth.astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    return d


def deproject(mask: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels in `mask` with valid depth -> (N, 3) points in the optical frame."""
    ys, xs = np.nonzero(mask & (depth_m > 0.15) & (depth_m < 6.0))
    z = depth_m[ys, xs]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1)


def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def object_centre(points_base: np.ndarray, height: float = OBJECT_HEIGHT, cap_band: float = 0.012):
    """Centre of a vertical cylinder from the visible surface points (base frame)."""
    if len(points_base) < 20:
        return None
    zmax = np.percentile(points_base[:, 2], 97)
    cap = points_base[points_base[:, 2] > zmax - cap_band]
    if len(cap) < 10:
        return None
    return np.array([np.median(cap[:, 0]), np.median(cap[:, 1]), np.median(cap[:, 2]) - height / 2])


def pad_centre(points_base: np.ndarray, thickness: float = PAD_THICKNESS):
    if len(points_base) < 20:
        return None
    z = np.median(points_base[:, 2])
    flat = points_base[np.abs(points_base[:, 2] - z) < 0.01]
    if len(flat) < 20:
        return None
    # the visible patch can be partly hidden by the object, so use the bounding-box centre
    lo, hi = np.percentile(flat[:, :2], 3, axis=0), np.percentile(flat[:, :2], 97, axis=0)
    return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z - thickness / 2])


def detect(rgb, depth_m, K, T_base_cam, obj_colour=BLUE, pad_colour=RED):
    """One frame -> (object_centre | None, pad_centre | None) in the base frame."""
    out = []
    for colour, fn in ((obj_colour, object_centre), (pad_colour, pad_centre)):
        m = clean_mask(colour.mask(rgb))
        if not m.any():
            out.append(None)
            continue
        pts = transform_points(deproject(m, depth_m, K), T_base_cam)
        out.append(fn(pts))
    return out[0], out[1]


class PerceptionNoise:
    """Corrupts a clean observation the way the camera pipeline does (for training and testing in sim).

    * a per-episode bias on the object and pad estimates (calibration / hand-eye error),
    * per-step jitter,
    * stale frames: with probability `hold_prob` the previous perception is reused,
    * finger-opening noise.
    The object estimate is left clean while it is held (the real robot uses hand pose then).
    """

    def __init__(self, seed=0, bias_xy=0.010, bias_z=0.007, jitter=0.004, hold_prob=0.1, finger=0.0005):
        self.rng = np.random.default_rng(seed + 777)
        self.bias_xy, self.bias_z = bias_xy, bias_z
        self.jitter, self.hold_prob, self.finger = jitter, hold_prob, finger
        self.new_episode()

    def new_episode(self):
        sig = np.array([self.bias_xy, self.bias_xy, self.bias_z])
        self.bias_obj = self.rng.normal(0, sig)
        self.bias_pad = self.rng.normal(0, sig)
        self._prev = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = obs.copy()
        attached = obs[11] > 0.5
        if self._prev is not None and self.rng.random() < self.hold_prob:
            out[3:9] = self._prev  # stale frame (a delay of one step)
        else:
            jo = self.rng.normal(0, self.jitter, 3)
            jp = self.rng.normal(0, self.jitter, 3)
            out[3:6] += 0 if attached else self.bias_obj + jo
            out[6:9] += self.bias_pad + jp
        self._prev = out[3:9].copy()
        out[9] += self.rng.normal(0, self.finger)
        return out
