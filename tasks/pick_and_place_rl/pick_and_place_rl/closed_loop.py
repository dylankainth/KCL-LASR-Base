"""Closed-loop helpers layered over the policy on the real robot (pure numpy, no ROS).

The camera and the joint encoders disagree by a roughly constant amount (head tilt, table height, calibration), and a
controller that only sees that disagreement can never remove it. So the loops here close on things that do see the
hand and the object together:

  HandTracker  encoder PI: the measured hand follows the commanded one (gravity droop, controller lag)
  GraspSearch  contact: a close that finds nothing retries on a small spiral; the offset that finally grips becomes
               a standing correction to the object estimate for the next episodes
  touchdown    contact: the hand has stopped although it is commanded down, so it rests on the pad / table

FingerSettle  the finger opening has stopped changing, so the grasp can be judged

Enabled with `env.closed_loop = True` (run_real --closed-loop). Everything is [unverified] on hardware.
"""

from collections import deque

import numpy as np

TRACK_KI = 0.15  # integral gain per control step on (commanded - measured) hand position
TRACK_LIMIT = 0.015  # m, largest correction the integral may add to a command
TRACK_SETTLED = 0.012  # m, integrate only while the hand is this close to the command (not during fast moves)
TRACK_LEAK = 0.98  # per step, so a stale correction fades
SEARCH_RINGS = (0.010, 0.020)  # m, radii of the retry spiral around the believed object position
SEARCH_POINTS = 8  # retry points per ring
BIAS_LIMIT = 0.03  # m, the standing object correction never exceeds this
SETTLE_STEPS = 3  # control steps of finger readings that must agree before a grasp is judged
SETTLE_TOL = 0.0006  # m, largest spread of those readings that still counts as 'the fingers stopped'
TOUCH_GAP = 0.004  # m, the hand is this far above a downward command ...
TOUCH_STILL = 0.001  # m, ... and has moved less than this over the last TOUCH_WINDOW steps: it is resting on something
TOUCH_WINDOW = 3  # control steps of hand height kept for that test


class HandTracker:
    """Adds a bounded integral of (commanded - measured) hand position to the next command."""

    def __init__(self, ki=TRACK_KI, limit=TRACK_LIMIT, settled=TRACK_SETTLED, leak=TRACK_LEAK):
        self.ki, self.limit, self.settled, self.leak = ki, limit, settled, leak
        self.i = np.zeros(3)

    def reset(self):
        self.i[:] = 0.0

    def correction(self, cmd, hand, hold=False):
        """`hold` freezes the integral (contact, gripper closing) so it cannot wind up against an obstacle."""
        err = np.asarray(cmd, float) - np.asarray(hand, float)
        if not hold and np.linalg.norm(err) < self.settled:
            self.i = np.clip(self.leak * self.i + self.ki * err, -self.limit, self.limit)
        return self.i.copy()


class FingerSettle:
    """Has the finger opening stopped changing? A close sweeps through the object's width on its way to the command
    value when nothing is in the way, so a grasp may only be judged once the fingers have stopped."""

    def __init__(self, steps=SETTLE_STEPS, tol=SETTLE_TOL):
        self.hist = deque(maxlen=steps)
        self.tol = tol

    def clear(self):
        self.hist.clear()

    def push(self, opening):
        self.hist.append(float(opening))

    def settled(self) -> bool:
        return len(self.hist) == self.hist.maxlen and max(self.hist) - min(self.hist) < self.tol


class GraspSearch:
    """Retry offsets after a grasp that closed on nothing, and the correction learnt from a grasp that worked."""

    def __init__(self, rings=SEARCH_RINGS, points=SEARCH_POINTS, bias_limit=BIAS_LIMIT):
        ang = np.arange(points) * 2 * np.pi / points
        self.offsets = [np.zeros(2)] + [r * np.array([np.cos(a), np.sin(a)]) for r in rings for a in ang]
        self.bias_limit = bias_limit
        self.bias = np.zeros(2)  # standing correction added to every object estimate (xy, base frame)
        self.k = 0  # index into offsets for the current attempt
        self.misses = 0  # total missed grasps (statistics)

    @property
    def offset(self) -> np.ndarray:
        """xyz to add to the perceived object position now (standing bias + this attempt's search offset)."""
        return np.array([*(self.bias + self.offsets[self.k]), 0.0])

    def new_episode(self):
        self.k = 0

    def missed(self) -> bool:
        """The fingers closed on nothing. Move to the next retry point; False when the spiral is used up."""
        self.misses += 1
        if self.k + 1 >= len(self.offsets):
            return False
        self.k += 1
        return True

    def grasped(self) -> np.ndarray:
        """The fingers gripped. The offset that worked becomes part of the standing correction."""
        self.bias = np.clip(self.bias + self.offsets[self.k], -self.bias_limit, self.bias_limit)
        self.k = 0
        return self.bias.copy()


def touchdown(cmd_z, hand_z, dz_cmd, moved, gap=TOUCH_GAP, still=TOUCH_STILL) -> bool:
    """True when the hand has stopped (`moved` = its height range over the last few steps) although the command is
    still `gap` below it and being driven down. Lag alone does not qualify: a hand that follows moves."""
    return dz_cmd < 0.0 and (hand_z - cmd_z) > gap and moved < still
