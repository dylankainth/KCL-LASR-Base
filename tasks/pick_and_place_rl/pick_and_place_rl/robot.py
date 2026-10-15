"""ROS 2 interface to TIAGo + the pick_place Gazebo scene."""

import threading
import time

import numpy as np
import rclpy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import GetEntityState, SetEntityState
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from pick_and_place_rl.kinematics import Chain

ARM_JOINTS = [f"arm_{i}_joint" for i in range(1, 8)]
TORSO_JOINT = "torso_lift_joint"
FINGER_JOINTS = ["gripper_right_finger_joint", "gripper_left_finger_joint"]
CHAIN_JOINTS = [TORSO_JOINT] + ARM_JOINTS
OBJECT_NAME = "pick_object"
PAD_NAME = "place_pad"
OCCLUSION_RADIUS = 0.15  # hand within this xy distance of the latched estimate -> ignore new detections
OCCLUSION_HEIGHT = 1.35  # ... and below this hand height (the camera sits ~1.3 m up)
MAX_JUMP = 0.35  # rad, refuse a servo command further than this from the measured state
ROUTE_JUMP = 3.5  # rad, limit for the scripted routes (PAL's unfold rotates the wrist ~1.6 rad); still speed-limited
GRASP_DZ = 0.06  # hand height above the object's centre when grasping (same as env.GRASP_DZ)
GRIPPER_OPEN = 0.045
GRIPPER_CLOSED = 0.0

# Gripper axes in the `gripper_grasping_frame` (measured from the URDF):
# approach direction = local +x, finger opening axis = local -y.
APPROACH_LOCAL = [1.0, 0.0, 0.0]
FINGER_LOCAL = [0.0, -1.0, 0.0]


def _duration(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int((seconds - sec) * 1e9))


class _TrajectoryOut:
    """Sends single-point trajectories to a PAL controller.

    Primary path: the `/<name>_controller/follow_joint_trajectory` action (the interface the rest of this repo
    uses on the real robot). A new goal replaces the previous one, so streaming small servo goals works.
    Fallback: the `/<name>_controller/joint_trajectory` topic, used when no action server answers."""

    def __init__(self, node, name, use_action=True, wait=3.0):
        self.name = name
        self.goals_sent = 0
        self.rejected = 0
        self.action = None
        if use_action:
            client = ActionClient(node, FollowJointTrajectory, f"/{name}_controller/follow_joint_trajectory")
            if client.wait_for_server(timeout_sec=wait):
                self.action = client
        self.pub = None if self.action else node.create_publisher(JointTrajectory, f"/{name}_controller/joint_trajectory", 10)

    @property
    def mode(self):
        return "action" if self.action else "topic"

    def _on_response(self, fut):
        try:
            if not fut.result().accepted:
                self.rejected += 1
        except Exception:
            self.rejected += 1

    def send(self, msg: JointTrajectory):
        self.goals_sent += 1
        if self.action:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = msg
            self.action.send_goal_async(goal).add_done_callback(self._on_response)
        else:
            self.pub.publish(msg)


class TiagoScene:
    """Joint state, arm/torso/gripper commands and ground-truth object poses."""

    def __init__(
        self,
        urdf_xml: str,
        node_name: str = "pick_place_rl",
        sim: bool = True,
        use_perception: bool = False,
        speed_scale: float = 1.0,
        max_joint_speed: float = 0.8,
        use_actions: bool = True,
    ):
        """sim=True: Gazebo (ground-truth poses, teleporting, grasp assist).
        sim=False: the real robot (poses from the perception node, no teleporting, no grasp assist,
        joint speed limits enforced on every command)."""
        self.sim = sim
        self.use_perception = use_perception or not sim  # object/pad poses from the camera pipeline
        self._perc = {}
        self._est_time = {}
        self.z_offset = 0.0  # real robot: table height minus 0.815 (set by the runner)
        self.calib = np.zeros(3)  # added to every camera estimate (base frame), from calibrate.py
        self.carrying = False  # real robot: the env set this when the fingers closed on the object
        self.max_joint_speed = max_joint_speed * speed_scale
        if not rclpy.ok():
            rclpy.init()
        self.node = Node(
            node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, sim)],
        )
        self.chain = Chain(urdf_xml, "base_footprint", "gripper_grasping_frame")
        assert self.chain.joint_names == CHAIN_JOINTS, self.chain.joint_names

        self._lock = threading.Lock()
        self._joints = {}
        self._models = {}
        self._yaws = {}
        self.node.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        if sim:
            self.node.create_subscription(
                ModelStates, "/gazebo/model_states", self._on_models, 10
            )
        if self.use_perception:
            self.node.create_subscription(
                PointStamped, "/pick_place/object", lambda m: self._on_point(OBJECT_NAME, m), 5
            )
            self.node.create_subscription(
                PointStamped, "/pick_place/pad", lambda m: self._on_point(PAD_NAME, m), 5
            )
            self.node.create_subscription(Float64MultiArray, "/pick_place/table", self._on_table, 5)
        self._table = None
        self._table_time = 0.0
        self._arm_pub = _TrajectoryOut(self.node, "arm", use_actions)
        self._torso_pub = _TrajectoryOut(self.node, "torso", use_actions)
        self._grip_pub = _TrajectoryOut(self.node, "gripper", use_actions)
        self._head_pub = _TrajectoryOut(self.node, "head", use_actions)
        self.command_modes = {o.name: o.mode for o in (self._arm_pub, self._torso_pub, self._grip_pub, self._head_pub)}
        self._attach_pub = self.node.create_publisher(String, "/grasp_attach/command", 10)
        if sim:
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
        print(f"[robot] command interfaces: {self.command_modes}", flush=True)

    # ------------------------------------------------------------------ state
    _last_joint_wall = 0.0

    def _on_joints(self, msg: JointState):
        self._last_joint_wall = time.time()
        with self._lock:
            self._joints.update(zip(msg.name, msg.position))

    def _on_models(self, msg: ModelStates):
        with self._lock:
            for name, pose in zip(msg.name, msg.pose):
                self._models[name] = np.array(
                    [pose.position.x, pose.position.y, pose.position.z]
                )
                q = pose.orientation
                self._yaws[name] = float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def _on_table(self, msg: Float64MultiArray):
        with self._lock:
            self._table = np.array(msg.data, float)
            self._table_time = time.time()

    def table_fit(self, max_age: float = 1.0):
        """Latest table height / edge fit from the perception node (TableFit) or None if stale."""
        from pick_and_place_rl.table import TableFit

        with self._lock:
            if self._table is None or time.time() - self._table_time > max_age:
                return None
            return TableFit.from_array(self._table)

    def joints_still(self, names, window=0.6, tol=0.002) -> bool:
        """True if none of the named joints moved more than tol (rad/m) over `window` seconds."""
        with self._lock:
            a = np.array([self._joints[n] for n in names])
        self.sleep(window)
        with self._lock:
            b = np.array([self._joints[n] for n in names])
        return bool(np.max(np.abs(b - a)) < tol)

    def clear_table(self):
        with self._lock:
            self._table = None

    def truth_pose2d(self, name: str):
        """Gazebo ground truth (simulation only): (x, y, yaw) in the world frame."""
        with self._lock:
            p = self._models[name]
            return float(p[0]), float(p[1]), self._yaws[name]

    def _on_point(self, name, msg: PointStamped):
        """Real robot: a perception estimate (base frame). Latched while the hand occludes the object."""
        p = np.array([msg.point.x, msg.point.y, msg.point.z])
        with self._lock:
            have_q = all(j in self._joints for j in CHAIN_JOINTS)
            q = np.array([self._joints[j] for j in CHAIN_JOINTS]) if have_q else None
        if q is not None:
            if name == OBJECT_NAME and self.carrying:
                return
            ref = self._perc.get(name)
            if ref is not None:
                hand = self.chain.fk(q)[:3, 3]
                if np.linalg.norm(hand[:2] - ref[:2]) < OCCLUSION_RADIUS and hand[2] < OCCLUSION_HEIGHT:
                    return  # the hand/can is between the camera and the target: keep the last clean estimate
        with self._lock:
            self._perc[name] = p
            self._est_time[name] = time.time()

    def wait_ready(self, timeout: float = 60.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                ok = all(j in self._joints for j in CHAIN_JOINTS + FINGER_JOINTS)
                if self.sim:
                    ok = ok and OBJECT_NAME in self._models
            if ok and (not self.sim or self._set_state.service_is_ready()):
                return
            time.sleep(0.2)
        raise TimeoutError("TIAGo not ready (joint_states / model_states / services)")

    def wait_perception(self, timeout: float = 30.0, max_age: float = 1.5):
        """Real robot: block until fresh object and pad estimates exist."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            now = time.time()
            with self._lock:
                ok = all(now - self._est_time.get(n, -1e9) < max_age for n in (OBJECT_NAME, PAD_NAME))
            if ok:
                return True
            time.sleep(0.1)
        return False

    def unlatch_pad(self):
        with self._lock:
            self._perc.pop(PAD_NAME, None)
            self._est_time.pop(PAD_NAME, None)

    def unlatch_object(self):
        """Real robot: forget the latched object estimate so the next detection is accepted."""
        with self._lock:
            self._perc.pop(OBJECT_NAME, None)
            self._est_time.pop(OBJECT_NAME, None)

    def now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def sleep(self, seconds: float):
        """Sleep in (simulation or wall) time."""
        t0 = self.now()
        while self.now() - t0 < seconds:
            time.sleep(0.002)
        if not self.sim:
            self.check_alive()

    def check_alive(self, max_age: float = 0.5):
        """Real robot watchdog: abort if joint states stopped arriving."""
        if self._last_joint_wall > 0 and time.time() - self._last_joint_wall > max_age:
            raise RuntimeError("joint_states stopped (robot driver down or e-stop pressed)")

    def q(self) -> np.ndarray:
        with self._lock:
            return np.array([self._joints[j] for j in CHAIN_JOINTS])

    def gripper_opening(self) -> float:
        """Mean finger joint position (0 closed, 0.045 fully open)."""
        with self._lock:
            return float(np.mean([self._joints[j] for j in FINGER_JOINTS]))

    def ee_pos(self, q=None) -> np.ndarray:
        return self.chain.fk(self.q() if q is None else q)[:3, 3]

    def truth_pos(self, name: str) -> np.ndarray:
        """Gazebo ground truth (simulation only)."""
        with self._lock:
            return self._models[name].copy()

    def model_pos(self, name: str) -> np.ndarray:
        """What the robot knows: ground truth in plain sim, camera estimates otherwise."""
        if not self.sim and name == "tiago":
            return np.zeros(3)  # the base frame is the reference; nothing to re-pin
        if self.use_perception and name in (OBJECT_NAME, PAD_NAME):
            if name == OBJECT_NAME and self.carrying:
                return self.ee_pos() + np.array([0.0, 0.0, -GRASP_DZ - self.z_offset])
            with self._lock:
                p = self._perc[name].copy() + self.calib
            p[2] -= self.z_offset
            return p
        return self.truth_pos(name)

    # --------------------------------------------------------------- commands
    def _send(self, pub, names, positions, duration):
        msg = JointTrajectory()
        msg.joint_names = list(names)
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        pt.time_from_start = _duration(duration)
        msg.points = [pt]
        pub.send(msg)

    def _limited_duration(self, names, target, duration, max_jump=MAX_JUMP):
        """Real robot: stretch the duration so no joint exceeds the speed limit; refuse jumps."""
        with self._lock:
            cur = np.array([self._joints[j] for j in names])
        d = np.abs(np.asarray(target, float) - cur)
        if d.max() > max_jump:
            raise RuntimeError(f"refusing a {d.max():.2f} rad jump on {names[int(d.argmax())]}")
        return max(duration, float(d.max() / self.max_joint_speed))

    def send_arm(self, q_arm, duration=0.15, max_jump=MAX_JUMP):
        if not self.sim:
            duration = self._limited_duration(ARM_JOINTS, q_arm, duration, max_jump)
        self._send(self._arm_pub, ARM_JOINTS, q_arm, duration)
        return duration  # the duration actually used (longer than asked if the speed limit applied)

    def send_torso(self, height, duration=1.0):
        if not self.sim:
            duration = max(duration, abs(height - self.q()[0]) / 0.05)  # torso <= 5 cm/s
        self._send(self._torso_pub, [TORSO_JOINT], [height], duration)
        return duration

    def send_head(self, pan, tilt, duration=2.0):
        self._send(self._head_pub, ["head_1_joint", "head_2_joint"], [pan, tilt], duration)

    def send_gripper(self, opening, duration=0.3):
        opening = float(np.clip(opening, GRIPPER_CLOSED, GRIPPER_OPEN))
        self._send(self._grip_pub, FINGER_JOINTS, [opening, opening], duration)

    # ------------------------------------------------------------------ grasp assist
    def attach_object(self):
        """Rigidly attach the object to the gripper (see gazebo_grasp_attach)."""
        self.carrying = True  # real robot: bookkeeping only, the fingers hold the object
        if not self.sim:
            return
        self._attach_pub.publish(String(data=f"attach tiago arm_7_link {OBJECT_NAME} link"))

    def detach_object(self):
        self.carrying = False
        if not self.sim:
            return
        self._attach_pub.publish(String(data=f"detach tiago arm_7_link {OBJECT_NAME} link"))

    # ------------------------------------------------------------------ scene
    def set_model_pos(self, name: str, pos, yaw: float = 0.0):
        if not self.sim:
            raise RuntimeError("cannot teleport objects on the real robot")
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
