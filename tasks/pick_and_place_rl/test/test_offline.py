"""Offline regression tests (numpy only, no ROS, no Gazebo).

    cd tasks/pick_and_place_rl && python3 -m unittest discover -s test -v

They run perception.py / table.py on the synthetic imperfect camera of synthetic_check.py and check the
scene sanity gate. They say nothing about the real camera; they stop the pipeline from silently getting worse.
"""

import sys
import types
import unittest
from dataclasses import replace

import numpy as np

from pick_and_place_rl import perception as P
from pick_and_place_rl import synthetic_check as S
from pick_and_place_rl.table import fit_table, points_from_depth

BASE = S.SCENARIOS[1]  # realistic depth noise


def one_frame(sc, seed, obj=(0.63, 0.15), pad=(0.63, -0.20)):
    rng = np.random.default_rng(seed)
    T = S.lookat(S.CAM_POS, S.LOOK_AT)
    rgb, depth, _ = S.render(sc, np.array(obj), np.array(pad), rng, T)
    dm, enc = S.degrade_depth(depth, sc, rng)
    return rgb, P.depth_to_metres(dm, enc), T


class PerceptionTests(unittest.TestCase):
    def test_object_estimate_is_not_biased(self):
        # the old median-of-cap-band estimator sat ~8 mm toward the camera on perfect data
        errs = []
        for seed in range(4):
            rgb, dm, T = one_frame(BASE, seed, obj=(0.60 + 0.02 * seed, 0.08 + 0.05 * seed))
            o, _ = P.detect(rgb, dm, S.K, T)
            self.assertIsNotNone(o)
            errs.append(o[:2] - np.array([0.60 + 0.02 * seed, 0.08 + 0.05 * seed]))
        self.assertLess(np.abs(np.mean(errs, axis=0)).max(), 0.004)
        self.assertLess(np.abs(errs).max(), 0.006)

    def test_pad_estimate(self):
        rgb, dm, T = one_frame(BASE, 1)
        _, p = P.detect(rgb, dm, S.K, T)
        self.assertIsNotNone(p)
        self.assertLess(np.linalg.norm(p[:2] - np.array([0.63, -0.20])), 0.006)

    def test_bigger_blue_blob_is_not_taken_for_the_can(self):
        rgb, dm, T = one_frame(replace(BASE, distractor=True), 2)
        o, p = P.detect(rgb, dm, S.K, T)
        self.assertIsNotNone(o)
        self.assertLess(np.linalg.norm(o[:2] - np.array([0.63, 0.15])), 0.01)
        self.assertLess(np.linalg.norm(p[:2] - np.array([0.63, -0.20])), 0.01)

    def test_dim_room_still_finds_both(self):
        rgb, dm, T = one_frame(replace(BASE, brightness=0.3, pixel_noise=6.0), 3)
        o, p = P.detect(rgb, dm, S.K, T)
        self.assertIsNotNone(o)
        self.assertIsNotNone(p)

    def test_coloured_light_and_glare(self):
        for sc in (replace(BASE, cast=(1.15, 1.0, 0.72)), replace(BASE, cast=(0.82, 1.0, 1.2)), replace(BASE, specular=True)):
            rgb, dm, T = one_frame(sc, 4)
            o, p = P.detect(rgb, dm, S.K, T)
            self.assertIsNotNone(o, sc)
            self.assertIsNotNone(p, sc)
            self.assertLess(np.linalg.norm(o[:2] - np.array([0.63, 0.15])), 0.008)

    def test_no_blue_thing_means_no_object(self):
        rgb, dm, T = one_frame(BASE, 5)
        rgb = rgb.copy()
        rgb[..., 2] = np.minimum(rgb[..., 2], rgb[..., 0])  # remove all blue dominance
        o, _ = P.detect(rgb, dm, S.K, T)
        self.assertIsNone(o)

    def test_table_fit(self):
        rgb, dm, T = one_frame(BASE, 6)
        fit, why = fit_table(points_from_depth(dm, S.K, T))
        self.assertIsNotNone(fit, why)
        self.assertLess(abs(fit.height - S.TABLE_Z), 0.01)
        self.assertLess(abs(fit.edge_x - S.EDGE_X), 0.015)
        self.assertLess(abs(np.degrees(fit.yaw_err)), 0.6)


def _stub_ros_modules():
    """env.py imports robot.py (rclpy) at the top; the sanity gate itself is pure, so stub those imports."""
    for name, attrs in (
        ("pick_and_place_rl.robot", {"OBJECT_NAME": "pick_object", "PAD_NAME": "place_pad", "TiagoScene": object}),
        ("pick_and_place_rl.arm_control", {"HOVER_Z": 1.0, "ArmController": object}),
    ):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__dict__.update(attrs)
            sys.modules[name] = m


class SceneGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _stub_ros_modules()
        from pick_and_place_rl import env

        cls.env = env

    def test_plausible_scene_passes(self):
        e = self.env
        self.assertEqual(e.scene_problems(np.array([0.63, 0.15, e.OBJ_Z]), np.array([0.63, -0.2, e.PAD_Z])), [])

    def test_wrong_table_height_is_caught(self):
        e = self.env
        bad = e.scene_problems(np.array([0.63, 0.15, e.OBJ_Z + 0.06]), np.array([0.63, -0.2, e.PAD_Z + 0.06]))
        self.assertEqual(len(bad), 2)

    def test_outside_region_is_caught(self):
        e = self.env
        bad = e.scene_problems(np.array([0.95, 0.45, e.OBJ_Z]), np.array([0.63, -0.2, e.PAD_Z]))
        self.assertEqual(len(bad), 1)
        self.assertIn("object", bad[0])


class ClosedLoopTests(unittest.TestCase):
    def test_tracker_removes_steady_droop_and_is_bounded(self):
        from pick_and_place_rl.closed_loop import HandTracker

        t = HandTracker()
        cmd, hand = np.array([0.6, 0.0, 1.0]), np.array([0.6, 0.0, 0.994])  # 6 mm sag
        for _ in range(40):
            c = t.correction(cmd, hand)
            hand = cmd + np.array([0, 0, -0.006]) + c  # the arm still sags, but we now ask for more
        self.assertLess(abs(hand[2] - cmd[2]), 0.001)
        far = HandTracker()
        for _ in range(100):
            far.correction(cmd, cmd - np.array([0, 0, 0.05]))  # not following: must not wind up
        self.assertEqual(float(np.abs(far.i).max()), 0.0)
        held = HandTracker()
        for _ in range(100):
            held.correction(cmd, cmd - np.array([0, 0, 0.005]), hold=True)
        self.assertEqual(float(np.abs(held.i).max()), 0.0)

    def test_search_visits_a_spiral_then_gives_up(self):
        from pick_and_place_rl.closed_loop import GraspSearch

        g = GraspSearch()
        self.assertEqual(g.offset.tolist(), [0.0, 0.0, 0.0])
        seen = [g.offset[:2].copy()]
        while g.missed():
            seen.append(g.offset[:2].copy())
        self.assertEqual(len(seen), 1 + 8 * 2)
        self.assertAlmostEqual(max(np.linalg.norm(s) for s in seen), 0.02)

    def test_working_offset_becomes_standing_correction(self):
        from pick_and_place_rl.closed_loop import GraspSearch

        g = GraspSearch()
        g.missed()
        off = g.offset[:2].copy()
        b = g.grasped()
        np.testing.assert_allclose(b, off)
        np.testing.assert_allclose(g.offset[:2], off)  # next episode starts from the learnt correction
        self.assertTrue(g.k == 0)

    def test_bias_is_clamped(self):
        from pick_and_place_rl.closed_loop import GraspSearch

        g = GraspSearch()
        for _ in range(10):
            g.k = 1 + 0  # ring point at +x, 1 cm
            g.grasped()
        self.assertLessEqual(float(np.abs(g.bias).max()), 0.03 + 1e-12)

    def test_touchdown_needs_a_downward_command_below_the_hand(self):
        from pick_and_place_rl.closed_loop import touchdown

        self.assertTrue(touchdown(0.80, 0.82, -0.005))
        self.assertFalse(touchdown(0.80, 0.805, -0.005))  # ordinary tracking lag
        self.assertFalse(touchdown(0.80, 0.82, 0.005))  # moving up


class ClosedLoopEnvTests(unittest.TestCase):
    """env._closed_loop_action against a fake scene/arm (the ROS parts are not exercised)."""

    @classmethod
    def setUpClass(cls):
        _stub_ros_modules()
        from pick_and_place_rl import env

        cls.env = env

    def make(self, hand_z, cmd_z, stale=False, attached=True):
        e = self.env
        o = object.__new__(e.PickPlaceEnv)
        pad = np.array([0.63, -0.2, e.PAD_Z])
        o.sc = types.SimpleNamespace(
            perception_stale=lambda name, **k: stale, model_pos=lambda name: pad.copy()
        )
        hand = np.array([0.63, -0.2, hand_z])
        cmd = np.array([0.63, -0.2, cmd_z])
        o.arm = types.SimpleNamespace(hand_pos=lambda: hand, cmd_hand_pos=lambda: cmd)
        o.attached, o._touch, o._stale = attached, 0, 0
        return o

    def test_stale_camera_holds_then_gives_up(self):
        o = self.make(1.0, 1.0, stale=True)
        a = o._closed_loop_action(np.array([0.01, 0.01, -0.01, 1.0]))
        self.assertEqual(a[:3].tolist(), [0, 0, 0])
        o._stale = self.env.STALE_HOLD_STEPS
        with self.assertRaises(RuntimeError):
            o._closed_loop_action(np.array([0.0, 0.0, 0.0, 1.0]))

    def test_touchdown_releases_only_after_sustained_contact_near_the_pad(self):
        e = self.env
        place_z = e.PAD_Z + 0.05 + e.GRASP_DZ
        o = self.make(place_z + 0.01, place_z - 0.01)  # hand blocked 2 cm above its command, near place height
        down = np.array([0.0, 0.0, -0.005, 1.0])
        for _ in range(e.TOUCH_STEPS - 1):
            self.assertEqual(o._closed_loop_action(down)[3], 1.0)
        self.assertEqual(o._closed_loop_action(down)[3], 0.0)

    def test_no_release_high_above_the_pad_even_with_lag(self):
        e = self.env
        o = self.make(1.0, 0.97)  # 3 cm lag but 15 cm above the place height: that is just a fast descent
        for _ in range(10):
            self.assertEqual(o._closed_loop_action(np.array([0.0, 0.0, -0.005, 1.0]))[3], 1.0)


if __name__ == "__main__":
    unittest.main()
