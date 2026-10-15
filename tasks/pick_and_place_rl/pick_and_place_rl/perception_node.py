"""ROS 2 node: head camera -> /pick_place/object and /pick_place/pad (PointStamped, base_footprint).

    python3 -m pick_and_place_rl.perception_node [--sim] [--rgb TOPIC --depth TOPIC --info TOPIC]

Defaults are the Gazebo topics; on the real robot pass the real topics (see the runbook) and omit --sim.
Each published point is the median of the last few valid detections, so single bad frames are ignored.
"""

import argparse
import collections
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from pick_and_place_rl import perception as P

BASE_FRAME = "base_footprint"


def image_to_numpy(msg: Image) -> np.ndarray:
    dt = {"rgb8": np.uint8, "bgr8": np.uint8, "16UC1": np.uint16, "32FC1": np.float32}[msg.encoding]
    itemsize = np.dtype(dt).itemsize
    ch = 3 if msg.encoding in ("rgb8", "bgr8") else 1
    arr = np.frombuffer(bytes(msg.data), dtype=dt).reshape(msg.height, msg.step // itemsize)
    arr = arr[:, : msg.width * ch]
    arr = arr.reshape(msg.height, msg.width, ch) if ch == 3 else arr
    if msg.encoding == "bgr8":
        arr = arr[..., ::-1]
    return np.ascontiguousarray(arr)


def transform_to_matrix(tf) -> np.ndarray:
    t, q = tf.transform.translation, tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class PerceptionNode(Node):
    def __init__(self, args):
        super().__init__(
            "pick_place_perception",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, args.sim)],
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.K = None
        self.rgb = None
        self.hist = {"object": collections.deque(maxlen=args.window), "pad": collections.deque(maxlen=args.window)}
        self.pub = {
            "object": self.create_publisher(PointStamped, "/pick_place/object", 10),
            "pad": self.create_publisher(PointStamped, "/pick_place/pad", 10),
        }
        self.create_subscription(CameraInfo, args.info, self._on_info, 1)
        self.create_subscription(Image, args.rgb, self._on_rgb, 1)
        self.create_subscription(Image, args.depth, self._on_depth, 1)
        self.stats = {"frames": 0, "object": 0, "pad": 0}
        self.every = args.every
        self._n = 0
        self.frame = args.camera_frame

    def _on_info(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)
        if not self.frame:
            self.frame = msg.header.frame_id

    def _on_rgb(self, msg):
        self.rgb = image_to_numpy(msg)

    def _on_depth(self, msg):
        self._n += 1
        if self.K is None or self.rgb is None or self._n % self.every:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, self.frame or msg.header.frame_id, rclpy.time.Time(), Duration(seconds=0.2)
            )
        except Exception as e:  # tf not ready yet
            self.get_logger().warn(f"no TF: {e}", throttle_duration_sec=5.0)
            return
        depth = P.depth_to_metres(image_to_numpy(msg), msg.encoding)
        if depth.shape != self.rgb.shape[:2]:
            self.get_logger().warn("rgb and depth sizes differ (use registered depth)", throttle_duration_sec=5.0)
            return
        obj, pad = P.detect(self.rgb, depth, self.K, transform_to_matrix(tf))
        self.stats["frames"] += 1
        for name, est in (("object", obj), ("pad", pad)):
            if est is None:
                continue
            self.stats[name] += 1
            self.hist[name].append(est)
            m = np.median(np.array(self.hist[name]), axis=0)
            out = PointStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = BASE_FRAME
            out.point.x, out.point.y, out.point.z = (float(v) for v in m)
            self.pub[name].publish(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="use simulation time")
    ap.add_argument("--rgb", default="/head_front_camera/rgb/image_raw")
    ap.add_argument("--depth", default="/head_front_camera/depth/image_raw")
    ap.add_argument("--info", default="/head_front_camera/rgb/camera_info")
    ap.add_argument("--camera-frame", default="", help="optical frame (default: from camera_info)")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--every", type=int, default=2, help="process every n-th depth frame")
    args, _ = ap.parse_known_args()
    rclpy.init()
    node = PerceptionNode(args)
    t0 = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.time() - t0 > 10:
                node.get_logger().info(f"detections: {node.stats}")
                t0 = time.time()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


def start_inprocess(sc, urdf: str, table_top: float = 0.815):
    """Sim helper: run the perception node inside this process and aim the head at the table."""
    import types

    from rclpy.executors import SingleThreadedExecutor

    from pick_and_place_rl.head import solve_head

    pn = PerceptionNode(
        types.SimpleNamespace(
            sim=True, window=3, every=1, camera_frame="",
            rgb="/head_front_camera/rgb/image_raw",
            depth="/head_front_camera/depth/image_raw",
            info="/head_front_camera/rgb/camera_info",
        )
    )
    ex = SingleThreadedExecutor()
    ex.add_node(pn)
    import threading

    threading.Thread(target=ex.spin, daemon=True).start()
    sc.send_torso(0.35, 2.0)
    sc.sleep(2.5)
    pan, tilt, _, _ = solve_head(urdf, 0.35, (0.63, 0.0, table_top + 0.03))
    sc.send_head(pan, tilt, 2.0)
    sc.sleep(3.0)
    return pn
