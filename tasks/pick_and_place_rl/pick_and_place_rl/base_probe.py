"""Does the base respond at low speeds? Sweeps turn and drive speeds on /cmd_vel and compares odometry with ground truth.

    python3 -m pick_and_place_rl.base_probe
Rotates/drives the base a few degrees / centimetres per step and returns it. Area around the base must be clear.
"""

import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter  # noqa: F401
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock


def yaw_of(q):
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def main():
    rclpy.init()
    node = Node("pick_place_base_probe")
    s = {"odom": None, "gt": None, "clock": 0.0}

    def on_odom(m):
        s["odom"] = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))

    def on_gt(m):
        s["gt"] = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))

    def on_clock(m):
        s["clock"] = m.clock.sec + m.clock.nanosec * 1e-9

    node.create_subscription(Odometry, "/mobile_base_controller/odom", on_odom, 10)
    node.create_subscription(Odometry, "/ground_truth_odom", on_gt, 10)
    node.create_subscription(Clock, "/clock", on_clock, qos_profile_sensor_data)
    pub = node.create_publisher(Twist, "/cmd_vel", 10)
    t0 = time.time()
    while time.time() - t0 < 3:
        rclpy.spin_once(node, timeout_sec=0.1)

    # real-time factor
    c0, w0 = s["clock"], time.time()
    while time.time() - w0 < 3:
        rclpy.spin_once(node, timeout_sec=0.1)
    print(f"real-time factor {(s['clock'] - c0) / (time.time() - w0):.2f}", flush=True)

    def run(vx, wz, sim_seconds):
        o0, g0, c0 = s["odom"], s["gt"], s["clock"]
        while s["clock"] - c0 < sim_seconds:
            m = Twist()
            m.linear.x, m.angular.z = vx, wz
            pub.publish(m)
            rclpy.spin_once(node, timeout_sec=0.05)
        for _ in range(10):
            pub.publish(Twist())
            rclpy.spin_once(node, timeout_sec=0.05)
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.05)
        o1, g1 = s["odom"], s["gt"]
        return (np.degrees(o1[2] - o0[2]), np.degrees(g1[2] - g0[2]), np.hypot(o1[0] - o0[0], o1[1] - o0[1]), np.hypot(g1[0] - g0[0], g1[1] - g0[1]))

    for w in (0.04, 0.08, 0.12, 0.2, -0.2, 0.12 * -1):
        od, gd, _, _ = run(0.0, w, 3.0)
        print(f"turn cmd {w:+.2f} rad/s x 3 sim-s (expected {np.degrees(w * 3):+.1f} deg): odom {od:+.2f} deg, ground truth {gd:+.2f} deg", flush=True)
    for v in (0.03, 0.06, 0.10, -0.10):
        _, _, od, gd = run(v, 0.0, 3.0)
        print(f"drive cmd {v:+.2f} m/s x 3 sim-s (expected {abs(v) * 3 * 100:.0f} cm): odom {od * 100:.1f} cm, ground truth {gd * 100:.1f} cm", flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
