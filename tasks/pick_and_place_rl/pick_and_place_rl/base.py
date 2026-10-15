"""Slow, closed-loop driving of the TIAGo base (differential drive: forward/back and turn on the spot).

Commands go to `/cmd_vel` (Twist, through PAL's twist_mux, so the joystick can still override them) and progress
is measured from the wheel odometry. Hard limits: 0.12 m/s, 0.30 rad/s, a travel budget, a watchdog on odometry
and a check that the base actually moves. The camera survey in park.py closes the loop (odometry alone is off
by up to 2x in Gazebo). A zero command is always sent at the end, also on errors.
"""

import os
import time

import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

MAX_V = 0.12  # m/s
MAX_W = 0.30  # rad/s
MIN_V = 0.05  # the simulated controller ignores commands below ~0.04 m/s and ~0.15 rad/s (dead zone); tune on the robot
MIN_W = 0.15
DT = 0.05
ODOM_TOPICS = ("/mobile_base_controller/odom", "/odom")


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class BaseDriver:
    def __init__(self, scene, topic="/cmd_vel", odom=None, travel_budget_m=0.6, turn_budget_rad=8.0):
        self.sc = scene
        self.pub = scene.node.create_publisher(Twist, topic, 10)
        self._odom = None
        self._odom_wall = 0.0
        names = {n for n, _ in scene.node.get_topic_names_and_types()}
        odom_topic = odom or next((t for t in ODOM_TOPICS if t in names), ODOM_TOPICS[0])
        scene.node.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.odom_topic = odom_topic
        self.travel_left = travel_budget_m
        self.turn_left = turn_budget_rad
        self.moved = 0.0
        self.debug = bool(os.environ.get("BASE_DEBUG"))

    def _on_odom(self, msg):
        q, p = msg.pose.pose.orientation, msg.pose.pose.position
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self._odom = (p.x, p.y, float(yaw))
        self._odom_wall = time.time()

    def wait_odom(self, timeout=5.0):
        t0 = time.time()
        while self._odom is None and time.time() - t0 < timeout:
            time.sleep(0.05)
        if self._odom is None:
            raise RuntimeError(f"no odometry on {self.odom_topic}")

    def _pose(self):
        if time.time() - self._odom_wall > 0.5:
            raise RuntimeError("odometry stopped; aborting base motion")
        return self._odom

    def stop(self):
        for _ in range(3):
            self.pub.publish(Twist())
            time.sleep(0.02)

    def _command(self, v, w):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self.pub.publish(msg)

    def turn(self, dyaw: float, tol=0.02):
        """Turn on the spot by dyaw radians (counter-clockwise positive)."""
        if abs(dyaw) < tol:
            return
        if abs(dyaw) > self.turn_left:
            raise RuntimeError(f"turn of {dyaw:.2f} rad exceeds the remaining turn budget {self.turn_left:.2f}")
        self.turn_left -= abs(dyaw)
        self.wait_odom()
        target = wrap(self._pose()[2] + dyaw)
        t_end = self.sc.now() + abs(dyaw) / (0.4 * MAX_W) + 20.0
        t_check = self.sc.now() + 3.0
        yaw0 = self._pose()[2]
        sign0 = np.sign(dyaw)
        boost, mark_t, mark_y = 0.0, self.sc.now(), yaw0
        try:
            while True:
                err = wrap(target - self._pose()[2])
                if abs(err) < tol or np.sign(err) != sign0:  # reached, or crossed the target (do not reverse)
                    break
                if self.sc.now() > t_end:
                    raise RuntimeError("turn timed out")
                if self.sc.now() - mark_t > 1.0:  # breakaway: static friction can swallow small commands
                    if abs(wrap(self._pose()[2] - mark_y)) < 0.01:
                        boost = min(boost + 0.05, MAX_W - MIN_W)
                    mark_t, mark_y = self.sc.now(), self._pose()[2]
                if self.sc.now() > t_check and boost >= MAX_W - MIN_W and abs(wrap(self._pose()[2] - yaw0)) < 0.004:
                    raise RuntimeError("the base is not turning even at full command (cmd_vel not accepted, brakes on, e-stop?)")
                w = sign0 * float(np.clip(1.2 * abs(err), MIN_W + boost, MAX_W))
                if self.debug:
                    print(f"      turn: t={self.sc.now():.2f} yaw={self._pose()[2]:+.4f} err={err:+.4f} cmd w={w:+.3f}", flush=True)
                self._command(0.0, w)
                self.sc.sleep(DT)
        finally:
            self.stop()

    def drive(self, dist: float, tol=0.01):
        """Drive straight by dist metres (negative = reverse)."""
        if abs(dist) < tol:
            return
        if abs(dist) > self.travel_left:
            raise RuntimeError(f"drive of {dist:.2f} m exceeds the remaining travel budget {self.travel_left:.2f}")
        self.travel_left -= abs(dist)
        self.wait_odom()
        x0, y0, yaw0 = self._pose()
        ux, uy = np.cos(yaw0), np.sin(yaw0)
        t_end = self.sc.now() + abs(dist) / (0.4 * MAX_V) + 20.0
        t_check = self.sc.now() + 3.0
        sign0 = np.sign(dist)
        boost, mark_t, mark_d = 0.0, self.sc.now(), 0.0
        try:
            while True:
                x, y, _ = self._pose()
                done = (x - x0) * ux + (y - y0) * uy
                rem = dist - done
                if abs(rem) < tol or np.sign(rem) != sign0:
                    break
                if self.sc.now() > t_end:
                    raise RuntimeError("drive timed out")
                if self.sc.now() - mark_t > 1.0:
                    if abs(done - mark_d) < 0.004:
                        boost = min(boost + 0.02, MAX_V - MIN_V)
                    mark_t, mark_d = self.sc.now(), done
                if self.sc.now() > t_check and boost >= MAX_V - MIN_V and abs(done) < 0.002:
                    raise RuntimeError("the base is not moving even at full command (cmd_vel not accepted? brakes on? e-stop?)")
                v = sign0 * float(np.clip(1.0 * abs(rem), MIN_V + boost, MAX_V))
                self._command(v, 0.0)
                self.sc.sleep(DT)
            self.moved += abs(dist)
        finally:
            self.stop()

    def move_relative(self, dx: float, dy: float):
        """Shift the base by (dx forward, dy left) in its own frame, ending with the same heading.

        A differential drive cannot slide sideways, so this is turn, drive, turn back; it drives backwards when
        that needs the smaller turn."""
        dist = float(np.hypot(dx, dy))
        if dist < 0.01:
            return
        if dx >= 0:
            psi, signed = float(np.arctan2(dy, dx)), dist
        else:
            psi, signed = float(np.arctan2(-dy, -dx)), -dist
        self.turn(psi)
        self.drive(signed)
        self.turn(-psi)
