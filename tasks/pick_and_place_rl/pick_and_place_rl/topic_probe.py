"""List base-command and odometry topics with their types and publisher/subscriber counts (read-only).

    python3 -m pick_and_place_rl.topic_probe
"""

import re
import time

import rclpy
from rclpy.node import Node

PATTERN = re.compile(r"cmd_vel|nav_vel|odom|twist|mobile_base|scan")


def main():
    rclpy.init()
    node = Node("pick_place_topic_probe")
    t0 = time.time()
    while time.time() - t0 < 3.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    for name, types in sorted(node.get_topic_names_and_types()):
        if PATTERN.search(name):
            print(f"{name}  {types}  pubs={node.count_publishers(name)} subs={node.count_subscribers(name)}", flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
