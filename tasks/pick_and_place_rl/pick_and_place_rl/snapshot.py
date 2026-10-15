"""Save camera snapshots (rgb, depth, intrinsics, camera pose in the base frame) for offline perception work.

    python3 -m pick_and_place_rl.snapshot --out data/snap --tilts -0.60 -0.75 -0.875 [--table-height 0.815]

Moves only the torso and head. Use it with the arm tucked.
"""

import argparse
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from pick_and_place_rl.perception_node import BASE_FRAME, image_to_numpy, transform_to_matrix
from pick_and_place_rl.robot import TiagoScene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="/ros_ws/scratch/tiago.urdf")
    ap.add_argument("--out", default="data/snap")
    ap.add_argument("--tilts", type=float, nargs="+", default=[-0.60, -0.75, -0.875])
    ap.add_argument("--pan", type=float, default=0.0)
    ap.add_argument("--sim", action="store_true", default=True)
    args = ap.parse_args()

    sc = TiagoScene(open(args.urdf).read())
    node = Node("snapshot_listener")
    buf = Buffer()
    TransformListener(buf, node)
    got = {}
    node.create_subscription(CameraInfo, "/head_front_camera/rgb/camera_info", lambda m: got.__setitem__("K", np.array(m.k).reshape(3, 3)), 1)
    node.create_subscription(Image, "/head_front_camera/rgb/image_raw", lambda m: got.__setitem__("rgb", (m.header.stamp, image_to_numpy(m))), 1)
    node.create_subscription(Image, "/head_front_camera/depth/image_raw", lambda m: got.__setitem__("depth", (m.header.stamp, image_to_numpy(m), m.encoding, m.header.frame_id)), 1)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sc.send_torso(0.35, 2.0)
    sc.sleep(3.0)
    for tilt in args.tilts:
        got.pop("depth", None)
        sc.send_head(args.pan, tilt, 2.0)
        sc.sleep(3.5)
        t0 = time.time()
        while "depth" not in got and time.time() - t0 < 10:
            time.sleep(0.1)
        stamp, depth, enc, frame = got["depth"]
        tf = buf.lookup_transform(BASE_FRAME, frame, rclpy.time.Time(), Duration(seconds=1.0))
        name = f"{args.out}_tilt{tilt:+.3f}.npz"
        np.savez(name, rgb=got["rgb"][1], depth=depth, enc=enc, K=got["K"], T=transform_to_matrix(tf), tilt=tilt, pan=args.pan)
        print("saved", name, depth.shape, enc, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
