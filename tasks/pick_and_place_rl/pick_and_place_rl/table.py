"""Table height and near-edge line from depth points in the base frame (numpy only).

The table top is the biggest horizontal layer of points between 0.5 and 1.2 m. Its near edge is found per
2 cm slice of y (the smallest x with a table-height point), and a line x = a + b*y is fitted robustly.
    edge_x   a   distance from base_footprint to the edge, straight ahead (m)
    slope    b   edge direction; the robot is square to the table when b = 0
    yaw_err      angle the robot must turn (counter-clockwise positive) to be square: -atan(b)
"""

from dataclasses import dataclass

import numpy as np

Z_BAND = (0.5, 1.2)
LAYER_TOL = 0.015  # points within this of the table height belong to the table top
Y_LIMIT = 0.35  # use the middle of the image only (the sides are the least accurate)
BIN = 0.02
MIN_SPAN = 0.30  # edge must be seen over at least this width
MAX_RMS = 0.010
INLIER_TOL = 0.006


@dataclass
class TableFit:
    height: float
    edge_x: float
    slope: float
    span: float
    rms: float
    n_bins: int

    @property
    def yaw_err(self) -> float:
        return float(-np.arctan(self.slope))

    def as_array(self) -> np.ndarray:
        return np.array([self.height, self.edge_x, self.slope, self.span, self.rms, self.n_bins], float)

    @staticmethod
    def from_array(a) -> "TableFit":
        return TableFit(float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), int(a[5]))


def table_height(points: np.ndarray):
    """Height of the largest horizontal layer in the z band, or None."""
    z = points[:, 2]
    z = z[(z > Z_BAND[0]) & (z < Z_BAND[1])]
    if len(z) < 500:
        return None
    edges = np.arange(Z_BAND[0], Z_BAND[1] + 0.01, 0.01)
    hist, _ = np.histogram(z, edges)
    k = int(np.argmax(hist))
    z0 = 0.5 * (edges[k] + edges[k + 1])
    near = z[np.abs(z - z0) < LAYER_TOL]
    return float(np.median(near)) if len(near) >= 500 else None


def _robust_line(y, x, rng, iters=150):
    best = None
    n = len(y)
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        if abs(y[i] - y[j]) < 0.1:
            continue
        b = (x[j] - x[i]) / (y[j] - y[i])
        a = x[i] - b * y[i]
        inl = np.abs(x - (a + b * y)) < INLIER_TOL
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 6:
        return None
    A = np.stack([np.ones(best.sum()), y[best]], 1)
    (a, b), *_ = np.linalg.lstsq(A, x[best], rcond=None)
    return a, b, best


def fit_table(points: np.ndarray, seed: int = 0):
    """points (N, 3) in the base frame -> (TableFit | None, reason string)."""
    h = table_height(points)
    if h is None:
        return None, "no horizontal table-height layer in view"
    top = points[np.abs(points[:, 2] - h) < LAYER_TOL]
    top = top[np.abs(top[:, 1]) < Y_LIMIT]
    if len(top) < 500:
        return None, "too few table points in the middle of the view"
    ys, xs = [], []
    edges = np.arange(-Y_LIMIT, Y_LIMIT + 1e-9, BIN)
    idx = np.digitize(top[:, 1], edges)
    for k in range(1, len(edges)):
        m = idx == k
        if m.sum() >= 15:
            ys.append(0.5 * (edges[k - 1] + edges[k]))
            xs.append(np.percentile(top[m, 0], 3))  # near edge, tolerant of a stray point
    if len(ys) < 10:
        return None, "table edge not visible across enough of the view"
    ys, xs = np.array(ys), np.array(xs)
    res = _robust_line(ys, xs, np.random.default_rng(seed))
    if res is None:
        return None, "edge points do not form a line"
    a, b, inl = res
    span = float(ys[inl].max() - ys[inl].min())
    rms = float(np.sqrt(np.mean((xs[inl] - (a + b * ys[inl])) ** 2)))
    if span < MIN_SPAN:
        return None, f"straight edge seen over only {span * 100:.0f} cm (need {MIN_SPAN * 100:.0f})"
    if inl.sum() < 0.6 * len(ys):
        return None, f"only {inl.sum()}/{len(ys)} edge slices agree on a straight line (curved or covered edge?)"
    if rms > MAX_RMS:
        return None, f"edge line is rough (rms {rms * 1000:.0f} mm)"
    return TableFit(h, float(a), float(b), span, rms, int(inl.sum())), "ok"


def points_from_depth(depth_m: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, stride: int = 3) -> np.ndarray:
    """Subsampled depth image -> (N, 3) points in the base frame."""
    d = depth_m[::stride, ::stride]
    ys, xs = np.nonzero((d > 0.3) & (d < 5.0))
    z = d[ys, xs]
    u, v = xs * stride, ys * stride
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    return pc @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
