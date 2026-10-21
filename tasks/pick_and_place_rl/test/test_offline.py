"""Offline regression tests (numpy only, no ROS, no Gazebo).

    cd tasks/pick_and_place_rl && python3 -m unittest discover -s test -v

They run perception.py / table.py on the synthetic imperfect camera of synthetic_check.py and check the
scene sanity gate. They say nothing about the real camera; they stop the pipeline from silently getting worse.
"""

import sys
import types
from collections import deque
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

    def test_touchdown_needs_a_resting_hand_below_a_downward_command(self):
        from pick_and_place_rl.closed_loop import touchdown

        self.assertTrue(touchdown(0.80, 0.806, -0.005, moved=0.0002))  # resting 6 mm above where it is told to be
        self.assertFalse(touchdown(0.80, 0.806, -0.005, moved=0.012))  # same gap but still travelling: just lag
        self.assertFalse(touchdown(0.80, 0.802, -0.005, moved=0.0))  # within the gap
        self.assertFalse(touchdown(0.80, 0.806, 0.005, moved=0.0))  # being driven up


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
        o.fingers = e.FingerSettle()
        o._hz = deque(maxlen=e.TOUCH_WINDOW)
        o.touchdowns = o.stale_holds = 0
        o.grip_cmd = attached  # carrying means the gripper is commanded closed
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
        o = self.make(place_z + 0.0065, place_z)  # resting 6.5 mm above its command: the old dead band
        down = np.array([0.0, 0.0, -0.005, 1.0])
        grips = [o._closed_loop_action(down)[3] for _ in range(e.TOUCH_WINDOW + e.TOUCH_STEPS + 2)]
        first = grips.index(0.0)
        self.assertGreaterEqual(first, e.TOUCH_WINDOW + e.TOUCH_STEPS - 2)  # needs a full window, then sustained contact
        self.assertTrue(all(g == 1.0 for g in grips[:first]))

    def test_a_hand_that_is_still_moving_down_is_not_touching(self):
        e = self.env
        place_z = e.PAD_Z + 0.05 + e.GRASP_DZ
        o = self.make(place_z + 0.02, place_z)
        for k in range(12):  # lagging 8 mm behind its command but descending 4.8 mm per step
            o.arm.hand_pos = (lambda z=place_z + 0.02 - 0.0048 * k: np.array([0.63, -0.2, z]))
            o.arm.cmd_hand_pos = (lambda z=place_z + 0.012 - 0.0048 * k: np.array([0.63, -0.2, z]))
            self.assertEqual(o._closed_loop_action(np.array([0.0, 0.0, -0.005, 1.0]))[3], 1.0)

    def test_no_release_high_above_the_pad_even_with_lag(self):
        e = self.env
        o = self.make(1.0, 0.97)  # 3 cm lag but 15 cm above the place height: that is just a fast descent
        for _ in range(10):
            self.assertEqual(o._closed_loop_action(np.array([0.0, 0.0, -0.005, 1.0]))[3], 1.0)


class ClosedLoopStepTests(unittest.TestCase):
    """env.step with an integrating command and an arm that sags: the PI must settle, not creep."""

    @classmethod
    def setUpClass(cls):
        _stub_ros_modules()
        from pick_and_place_rl import env

        cls.env = env

    def run_steps(self, closed_loop, sag, n=120):
        e = self.env

        class Arm:
            def __init__(self):
                self.cmd = np.array([0.62, 0.0, 1.0])

            def cmd_hand_pos(self):
                return self.cmd.copy()

            def hand_pos(self):
                return self.cmd - sag

            def servo(self, target, dt):
                self.cmd = np.asarray(target, float).copy()

        pad = np.array([0.62, -0.2, e.PAD_Z])
        obj = np.array([0.62, 0.15, e.OBJ_Z])
        sc = types.SimpleNamespace(
            sim=False, gripper_opening=lambda: 0.045, send_gripper=lambda *a, **k: None,
            model_pos=lambda name: (obj if name == "pick_object" else pad).copy(),
            perception_stale=lambda name, **k: False, attach_object=lambda: None, detach_object=lambda: None,
            unlatch_object=lambda: None,
        )
        env = object.__new__(e.PickPlaceEnv)
        env.sc, env.arm = sc, Arm()
        env.step_scale, env.closed_loop = 1.0, closed_loop
        env.tracker, env.search, env.fingers = e.HandTracker(), e.GraspSearch(), e.FingerSettle()
        env._hz = deque(maxlen=e.TOUCH_WINDOW)
        env.touchdowns = env.stale_holds = 0
        env.grip_cmd = env.attached = False
        env._touch = env._stale = env._stall = 0
        env._stall_ref = None
        env._corr = np.zeros(3)
        env.pin_base, env.no_close, env.lost_count, env.steps, env.max_steps = False, False, 0, 0, 10**6
        env.search_exhausted, env.grip_close_cmd, env.obs_noise = False, e.GRIP_CLOSE_CMD, None
        env._check_following = lambda target: None
        hold = np.array([0.0, 0.0, 0.0, 0.0])
        for _ in range(n):
            env.step(hold)  # the policy asks for no motion: the hand should just sit where it is told to
        return env.arm.cmd, env.arm.hand_pos(), env._corr

    def test_pi_cancels_constant_sag_without_creeping(self):
        sag = np.array([0.0, 0.0, 0.008])
        cmd, hand, corr = self.run_steps(True, sag)
        self.assertLess(abs(hand[2] - 1.0), 0.0015)  # the hand reached where it was asked to be
        self.assertLess(np.abs(corr).max(), 0.0155)  # bounded
        cmd2, _, _ = self.run_steps(True, sag, n=400)
        self.assertLess(abs(cmd2[2] - cmd[2]), 0.001)  # and the command is not still drifting

    def test_without_closed_loop_the_sag_stays(self):
        _, hand, _ = self.run_steps(False, np.array([0.0, 0.0, 0.008]))
        self.assertAlmostEqual(hand[2], 1.0 - 0.008, places=6)


class GraspJudgementTests(unittest.TestCase):
    """Closed-loop grasp judgement on scripted finger openings (the part Gazebo cannot show: its fingers do not
    stop on the can)."""

    @classmethod
    def setUpClass(cls):
        _stub_ros_modules()
        from pick_and_place_rl import env

        cls.env = env

    def build(self, openings, offset_k=0):
        e = self.env
        holder = {}
        obj = np.array([0.62, 0.15, e.OBJ_Z])
        pad = np.array([0.62, -0.2, e.PAD_Z])
        sent = []
        moves = holder["moves"] = []

        class Arm:
            cmd = np.array([0.62, 0.15, e.OBJ_Z + e.GRASP_DZ])

            def cmd_hand_pos(self):
                return self.cmd.copy()

            def hand_pos(self):
                return self.cmd.copy()

            def servo(self, target, dt):
                self.cmd = np.asarray(target, float).copy()

            def move_to(self, target, **k):
                moves.append(np.asarray(target, float))
                self.cmd = np.asarray(target, float).copy()

        sc = types.SimpleNamespace(
            sim=False, send_gripper=lambda v, *a, **k: sent.append(v),
            sleep=lambda s: None, perception_ages=lambda: {"pick_object": 9.9, "place_pad": 0.1},
            wait_perception=lambda **k: holder["sees_seq"].pop(0) if holder.get("sees_seq") else holder.get("sees", True),
            gripper_opening=lambda: openings[min(holder["env"].steps, len(openings) - 1)],
            model_pos=lambda name: (obj if name == "pick_object" else pad).copy(),
            perception_stale=lambda name, **k: False, attach_object=lambda: None, detach_object=lambda: None,
            unlatch_object=lambda: None,
        )
        env = object.__new__(e.PickPlaceEnv)
        holder["env"] = env
        env.sc, env.arm = sc, Arm()
        env.step_scale, env.closed_loop = 1.0, True
        env.tracker, env.search, env.fingers = e.HandTracker(), e.GraspSearch(), e.FingerSettle()
        env._hz = deque(maxlen=e.TOUCH_WINDOW)
        env.touchdowns = env.stale_holds = 0
        env.search.k = offset_k
        env.grip_cmd = env.attached = False
        env._touch = env._stale = env._stall = 0
        env._stall_ref = None
        env._corr = np.zeros(3)
        env.pin_base, env.no_close, env.lost_count, env.steps, env.max_steps = False, False, 0, 0, 10**6
        env.search_exhausted, env.grip_close_cmd, env.obs_noise = False, 0.016, None
        env._check_following = lambda target: None
        env.holder = holder
        return env, sent

    def drive(self, env, n):
        """Run the real expert against the scripted finger openings."""
        history, k0 = [], env.search.k
        for _ in range(n):
            env.step(self.env.expert_action(env.obs()))  # the real expert, so its reopen-on-empty rule is in the loop
            history.append((env.attached, env.grip_cmd, env.search.k))
            if env.search.k != k0:
                break  # a miss: the real expert would reopen and approach again, not keep closing
        return history

    def test_stall_on_the_can_is_credited_only_after_the_fingers_settle(self):
        seq = [0.045, 0.040, 0.034, 0.029, 0.0258, 0.0252, 0.0251, 0.0251, 0.0251, 0.0251, 0.0251]
        env, _ = self.build(seq, offset_k=1)  # the first retry point was the one that worked
        off = env.search.offsets[1].copy()
        hist = self.drive(env, len(seq))
        first = next(i for i, h in enumerate(hist) if h[0])
        self.assertGreaterEqual(first, 6)  # not on the sweep sample at 0.0258
        self.assertTrue(all(h[1] for h in hist))  # never reopened
        np.testing.assert_allclose(env.search.bias, off)

    def test_empty_close_is_a_miss_and_never_credited(self):
        seq = [0.045, 0.040, 0.034, 0.029, 0.0258, 0.0206, 0.016, 0.016, 0.016, 0.016]
        env, sent = self.build(seq)
        hist = self.drive(env, 10)
        self.assertFalse(any(h[0] for h in hist))  # the sweep through 0.0258 must not attach
        self.assertEqual(env.search.k, 1)  # moved on to the next retry point
        self.assertEqual(float(np.abs(env.search.bias).max()), 0.0)
        self.assertEqual(sent[-1], self.env.GRIP_OPEN_CMD)  # reopened

    def test_a_miss_retreats_out_of_the_occlusion_zone_before_the_next_try(self):
        seq = [0.045, 0.040, 0.034, 0.029, 0.0258, 0.0206, 0.016, 0.016, 0.016, 0.016]
        env, _ = self.build(seq)
        self.drive(env, 10)
        moves = env.holder["moves"]
        self.assertEqual(len(moves), 2)
        self.assertAlmostEqual(moves[0][2], self.env.HOVER_Z)  # straight up first
        self.assertGreater(np.linalg.norm(moves[1][:2] - np.array([0.62, 0.15])), 0.17)  # then clear of the can
        self.assertAlmostEqual(moves[1][2], self.env.HOVER_Z)
        self.assertEqual(float(np.abs(env._corr).max()), 0.0)

    def test_if_one_side_hides_the_object_the_other_side_is_tried(self):
        seq = [0.045, 0.040, 0.034, 0.029, 0.0258, 0.0206, 0.016, 0.016, 0.016, 0.016]
        env, _ = self.build(seq)
        env.holder["sees_seq"] = [False, True]
        self.drive(env, 10)
        moves = env.holder["moves"]
        self.assertEqual(len(moves), 3)  # up, first side, other side
        self.assertNotEqual(round(float(moves[1][1]), 3), round(float(moves[2][1]), 3))

    def test_losing_the_object_after_a_miss_aborts(self):
        seq = [0.045, 0.040, 0.034, 0.029, 0.0258, 0.0206, 0.016, 0.016, 0.016, 0.016]
        env, _ = self.build(seq)
        env.holder["sees"] = False
        with self.assertRaises(RuntimeError) as cm:
            self.drive(env, 10)
        self.assertIn("find the object again", str(cm.exception))

    def test_the_hold_at_the_object_width_is_bounded(self):
        env, _ = self.build([0.025] * 80)
        env.grip_cmd = True  # fingers stalled at the can's width but the grasp is never accepted
        reopen = [env._closed_loop_action(np.array([0.0, 0.0, 0.0, 0.0]))[3] for _ in range(self.env.GRIP_WAIT_STEPS + 5)]
        self.assertTrue(all(g == 1.0 for g in reopen[: self.env.GRIP_WAIT_STEPS]))
        self.assertTrue(all(g == 0.0 for g in reopen[self.env.GRIP_WAIT_STEPS :]))  # then the expert may reopen

    def test_close_command_that_cannot_show_a_miss_is_refused(self):
        env, _ = self.build([0.045])
        env.grip_close_cmd = 0.0245
        env.fingers = self.env.FingerSettle()
        env.sc.wait_perception = lambda **k: True
        env.sc.unlatch_pad = lambda: None
        env.sc.sleep = lambda s: None
        env.arm.go_ready = lambda **k: None
        env._arm_ready = True
        env.arm.move_to = lambda *a, **k: None
        env.sc.send_gripper = lambda *a, **k: None
        with self.assertRaises(RuntimeError) as cm:
            env._reset_real(np.array([0.62, 0.0]))
        self.assertIn("grip-close", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
