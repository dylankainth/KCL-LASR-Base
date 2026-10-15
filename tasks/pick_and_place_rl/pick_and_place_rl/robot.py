"""ROS 2 interface to TIAGo + the pick_place Gazebo scene."""

import threading
import time

import numpy as np
import rclpy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import GetEntityState, SetEntityState
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from pick_and_place_rl.kinematics import Chain

ARM_JOINTS = [f"arm_{i}_joint" for i in range(1, 8)]
TORSO_JOINT = "torso_lift_joint"
FINGER_JOINTS = ["gripper_right_finger_joint", "gripper_left_finger_joint"]
CHAIN_JOINTS = [TORSO_JOINT] + ARM_JOINTS
OBJECT_NAME = "pick_object"
PAD_NAME = "place_pad"
GRIPPER_OPEN = 0.045
GRIPPER_CLOSED = 0.0

# Gripper axes in the `gripper_grasping_frame` (measured from the URDF):
# approach direction = local +x, finger opening axis = local -y.
APPROACH_LOCAL = [1.0, 0.0, 0.0]
FINGER_LOCAL = [0.0, -1.0, 0.0]


def _duration(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int((seconds - sec) * 1e9))


class TiagoScene:
    """Joint state, arm/torso/gripper commands and ground-truth object poses."""

    def __init__(self, urdf_xml: str, node_name: str = "pick_place_rl"):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node(
            node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.chain = Chain(urdf_xml, "base_footprint", "gripper_grasping_frame")
        assert self.chain.joint_names == CHAIN_JOINTS, self.chain.joint_names

        self._lock = threading.Lock()
        self._joints = {}
        self._models = {}
        self.node.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self.node.create_subscription(
            ModelStates, "/gazebo/model_states", self._on_models, 10
        )
        self._arm_pub = self.node.create_publisher(
            JointTrajectory, "/arm_controller/joint_trajectory", 10
        )
        self._torso_pub = self.node.create_publisher(
            JointTrajectory, "/torso_controller/joint_trajectory", 10
        )
        self._grip_pub = self.node.create_publisher(
            JointTrajectory, "/gripper_controller/joint_trajectory", 10
        )
        self._attach_pub = self.node.create_publisher(String, "/grasp_attach/command", 10)
        self._set_state = self.node.create_client(
            SetEntityState, "/gazebo/set_entity_state"
        )
        self._get_state = self.node.create_client(
            GetEntityState, "/gazebo/get_entity_state"
        )

        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self.wait_ready()

    # ------------------------------------------------------------------ state
    def _on_joints(self, msg: JointState):
        with self._lock:
            self._joints.update(zip(msg.name, msg.position))

    def _on_models(self, msg: ModelStates):
        with self._lock:
            for name, pose in zip(msg.name, msg.pose):
                self._models[name] = np.array(
                    [pose.position.x, pose.position.y, pose.position.z]
                )

    def wait_ready(self, timeout: float = 60.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                ok = all(j in self._joints for j in CHAIN_JOINTS + FINGER_JOINTS)
                ok = ok and OBJECT_NAME in self._models
            if ok and self._set_state.service_is_ready():
                return
            time.sleep(0.2)
        raise TimeoutError("TIAGo scene not ready (joint_states / model_states / services)")

    def now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def sleep(self, seconds: float):
        """Sleep in *simulation* time."""
        t0 = self.now()
        while self.now() - t0 < seconds:
            time.sleep(0.002)

    def q(self) -> np.ndarray:
        with self._lock:
            return np.array([self._joints[j] for j in CHAIN_JOINTS])

    def gripper_opening(self) -> float:
        """Mean finger joint position (0 closed, 0.045 fully open)."""
        with self._lock:
            return float(np.mean([self._joints[j] for j in FINGER_JOINTS]))

    def ee_pos(self, q=None) -> np.ndarray:
        return self.chain.fk(self.q() if q is None else q)[:3, 3]

    def model_pos(self, name: str) -> np.ndarray:
        with self._lock:
            return self._models[name].copy()

    # --------------------------------------------------------------- commands
    def _send(self, pub, names, positions, duration):
        msg = JointTrajectory()
        msg.joint_names = list(names)
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        pt.time_from_start = _duration(duration)
        msg.points = [pt]
        pub.publish(msg)

    def send_arm(self, q_arm, duration=0.15):
        self._send(self._arm_pub, ARM_JOINTS, q_arm, duration)

    def send_torso(self, height, duration=1.0):
        self._send(self._torso_pub, [TORSO_JOINT], [height], duration)

    def send_gripper(self, opening, duration=0.3):
        opening = float(np.clip(opening, GRIPPER_CLOSED, GRIPPER_OPEN))
        self._send(self._grip_pub, FINGER_JOINTS, [opening, opening], duration)

    # ------------------------------------------------------------------ grasp assist
    def attach_object(self):
        """Rigidly attach the object to the gripper (see gazebo_grasp_attach)."""
        self._attach_pub.publish(String(data=f"attach tiago arm_7_link {OBJECT_NAME} link"))

    def detach_object(self):
        self._attach_pub.publish(String(data=f"detach tiago arm_7_link {OBJECT_NAME} link"))

    # ------------------------------------------------------------------ scene
    def set_model_pos(self, name: str, pos, yaw: float = 0.0):
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose.position.x = float(pos[0])
        req.state.pose.position.y = float(pos[1])
        req.state.pose.position.z = float(pos[2])
        req.state.pose.orientation.z = float(np.sin(yaw / 2))
        req.state.pose.orientation.w = float(np.cos(yaw / 2))
        req.state.reference_frame = "world"
        fut = self._set_state.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 5.0:
            time.sleep(0.005)
        return fut.done() and fut.result().success

    def shutdown(self):
        self._executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()
