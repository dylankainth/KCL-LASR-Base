"""Read-only checks before moving the real robot. Publishes nothing to any controller.

    python3 -m pick_and_place_rl.preflight [--urdf FILE]

Checks: required joints/topics exist, the URDF kinematics match TF (so the IK is right for THIS robot),
camera topics + TF, and (if the perception node is running) fresh object / pad estimates.
Exit code 0 only if every hard check passes.
"""

import argparse
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from pick_and_place_rl.kinematics import Chain
from pick_and_place_rl.robot import CHAIN_JOINTS, FINGER_JOINTS

NEED_TOPICS = [
    "/joint_states",
    "/arm_controller/joint_trajectory",
    "/torso_controller/joint_trajectory",
    "/gripper_controller/joint_trajectory",
    "/head_controller/joint_trajectory",
]
results = []


def check(name, ok, detail="", hard=True):
    results.append((name, ok, hard))
    print(f"[{'PASS' if ok else ('FAIL' if hard else 'WARN')}] {name}" + (f" - {detail}" if detail else ""), flush=True)


def fetch_urdf(node, timeout=5.0):
    for srv in ("/robot_state_publisher/get_parameters",):
        cli = node.create_client(GetParameters, srv)
        if not cli.wait_for_service(timeout_sec=timeout):
            continue
        req = GetParameters.Request(names=["robot_description"])
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
        if fut.done() and fut.result().values:
            return fut.result().values[0].string_value
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=None, help="URDF file (default: robot_state_publisher's robot_description)")
    args = ap.parse_args()
    rclpy.init()
    node = Node("pick_place_preflight")
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    joints, pts = {}, {}
    node.create_subscription(JointState, "/joint_states", lambda m: joints.update(zip(m.name, m.position)), 10)
    for n in ("object", "pad"):
        node.create_subscription(PointStamped, f"/pick_place/{n}", lambda m, n=n: pts.__setitem__(n, (m, time.time())), 5)
    t0 = time.time()
    while time.time() - t0 < 3.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    # topics
    topics = dict(node.get_topic_names_and_types())
    for t in NEED_TOPICS:
        check(f"topic {t}", t in topics)
    cams = sorted(t for t in topics if ("image_raw" in t or "camera_info" in t) and "head" in t)
    print("      camera-like topics:", *cams, sep="\n        ")

    # joints
    miss = [j for j in CHAIN_JOINTS + FINGER_JOINTS if j not in joints]
    check("joint_states has torso/arm/finger joints", not miss, f"missing {miss}" if miss else "")
    if joints:
        print("      arm q:", np.round([joints.get(j, np.nan) for j in CHAIN_JOINTS], 3))
        print("      fingers:", np.round([joints.get(j, np.nan) for j in FINGER_JOINTS], 4))

    # kinematics vs TF (the single most important check: the IK assumes the URDF matches the robot)
    urdf = open(args.urdf).read() if args.urdf else fetch_urdf(node)
    check("robot_description available", urdf is not None)
    if urdf and not miss:
        chain = Chain(urdf, "base_footprint", "gripper_grasping_frame")
        check("chain joints are torso + arm_1..7", chain.joint_names == CHAIN_JOINTS)
        q = np.array([joints[j] for j in CHAIN_JOINTS])
        t_end = time.time() + 5
        tf = None
        while time.time() < t_end and tf is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                tf = tf_buffer.lookup_transform("base_footprint", "gripper_grasping_frame", rclpy.time.Time())
            except Exception:
                pass
        if tf is None:
            check("TF base_footprint -> gripper_grasping_frame", False)
        else:
            p_tf = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
            p_fk = chain.fk(q)[:3, 3]
            err = np.linalg.norm(p_tf - p_fk)
            check("URDF FK matches TF", err < 0.01, f"error {err * 1000:.1f} mm (fk {p_fk.round(3)}, tf {p_tf.round(3)})")
        for lo, hi, name in zip(chain.lower, chain.upper, chain.joint_names):
            if not (lo < joints[name] < hi):
                check(f"{name} inside URDF limits", False, f"{joints[name]:.3f} not in [{lo:.2f}, {hi:.2f}]")

    # camera TF
    ok_cam = False
    for frame in ("head_front_camera_color_optical_frame", "head_front_camera_rgb_optical_frame"):
        try:
            tf_buffer.lookup_transform("base_footprint", frame, rclpy.time.Time())
            ok_cam = True
            print(f"      camera frame: {frame}")
            break
        except Exception:
            pass
    check("TF base_footprint -> head camera optical frame", ok_cam, hard=False)

    # perception
    for n in ("object", "pad"):
        fresh = n in pts and time.time() - pts[n][1] < 2.0
        det = pts[n][0].point if n in pts else None
        check(
            f"perception publishes /pick_place/{n}",
            fresh,
            f"({det.x:.3f}, {det.y:.3f}, {det.z:.3f}) m" if det else "run perception_node first",
            hard=False,
        )
    hard_fail = [n for n, ok, hard in results if hard and not ok]
    print("\nPREFLIGHT", "FAILED: " + ", ".join(hard_fail) if hard_fail else "OK (hard checks passed)")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
