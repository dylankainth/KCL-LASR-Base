"""Numpy inference for the learned MLP policy (no torch needed inside the container)."""

import numpy as np

from pick_and_place_rl.env import MAX_STEP


class MLPPolicy:
    """obs -> (dx, dy, dz, grip). Weights come from `train_bc.py` (an .npz file)."""

    def __init__(self, weights: dict):
        self.obs_mean = weights["obs_mean"]
        self.obs_std = weights["obs_std"]
        self.layers = []
        i = 0
        while f"w{i}" in weights:
            self.layers.append((weights[f"w{i}"], weights[f"b{i}"]))
            i += 1

    @classmethod
    def load(cls, path):
        return cls(dict(np.load(path)))

    def act(self, obs) -> np.ndarray:
        x = (np.asarray(obs, np.float32) - self.obs_mean) / self.obs_std
        for k, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if k < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        delta = np.tanh(x[:3]) * MAX_STEP
        grip = 1.0 if x[3] > 0.0 else 0.0
        return np.array([*delta, grip])
