# Running pick_and_place_rl on the real TIAGo

Read this top to bottom before powering anything. **Everything here was verified in Gazebo only. Nothing has
touched the real robot.** Items marked **[unverified]** are assumptions that the preflight or the first trial
must confirm.

## 0. Safety

* Someone stands at the e-stop for every trial. The software limits (joint speed, jump check, workspace clamp)
  are a second line of defence, not the first.
* The arm moves at 30 % speed and the hand at 40 % of the trained step by default (`--speed-scale`,
  `--step-scale`). Raise them only after clean runs.
* Clear the area in front of the robot. The arm unfolds from the tucked pose and passes over the table
  (the route is PAL's `home` motion reversed, then up and over; it assumes the table edge is >= 0.55 m in front
  of the base).
* The first run uses the **scripted expert** (`--expert`), not the learned policy, so any failure is the
  hardware/perception, not the network.

## 1. What you need

| Item | Spec |
| :--- | :--- |
| Object | cylinder, **~5 cm diameter**, ~10 cm tall, light (<= 200 g), grippy, **saturated blue**. A tin can with a blue sleeve is fine. The policy was trained on r = 2.5 cm; a different diameter changes the finger opening the grasp logic expects (`CLOSED_ON_OBJ`, `GRIP_CLOSE_CMD` in `env.py`). |
| Pad | flat, **red**, ~14 x 14 cm (paper/card is fine), on the table |
| Table | top height measured in the robot base frame (tape measure from the floor, TIAGo base_footprint is on the floor). The sim table is 0.815 m; pass the real value with `--table-height`. |
| Placement | object in x 0.58-0.68 m, y +0.05..+0.28 m (left); pad x 0.58-0.68, y -0.28..-0.08 (right), measured from the base_footprint. Mark the regions on the table. The policy has only seen these. |
| Robot | TIAGo (PAL gripper, not the Hey5 hand) **[unverified]**, base parked square to the table about 0.55-0.60 m from its edge, torso lifts to 0.35 m |

## 2. Install

On the machine that talks to the robot (same container image as the sim, or the robot's ROS 2 Humble workspace):

```bash
cd ~/ros_ws && colcon build --packages-select pick_and_place_rl && source install/setup.bash
```

Only numpy and rclpy are needed at run time (no torch).

## 3. Bring-up order

```bash
# 1. read-only checks: topics, joints, URDF-vs-TF kinematics, camera TF. Moves nothing.
python3 -m pick_and_place_rl.preflight
```

Fix every `FAIL`. The most important one is `URDF FK matches TF` (< 10 mm): if it fails the IK is wrong for this
robot and nothing else matters. The preflight also prints the camera-like topics it finds.

```bash
# 2. perception (check the topic names printed by the preflight; defaults are the simulation names)
python3 -m pick_and_place_rl.perception_node \
    --rgb /head_front_camera/rgb/image_raw \
    --depth /head_front_camera/depth/image_raw \
    --info /head_front_camera/rgb/camera_info
# depth must be registered to the RGB image (same size); both 16UC1 (mm) and 32FC1 (m) are handled.
ros2 topic echo /pick_place/object   # base_footprint frame; compare with a tape measure
```

Tape-measure check: place the object at a known (x, y) from the base_footprint and compare. The sim error is
0.5 cm; on the real camera expect more. If the colour is not detected, change `BLUE` / `RED` in
`perception.py` (hue range in degrees, minimum saturation/value). **[unverified]**

```bash
# 3. first motion: scripted expert, slow, one episode
python3 -m pick_and_place_rl.run_real --expert --episodes 1 --table-height 0.80 \
    --speed-scale 0.2 --step-scale 0.3
# 4. the learned policy
python3 -m pick_and_place_rl.run_real --policy models/policy.npz --episodes 5 --table-height 0.80
```

`run_real` asks before moving, aims the head at the table, shows what it perceives, and asks again before every
episode. Ctrl-C holds the arm in place.

## 4. What will most likely need tuning (in order)

1. **Controller / topic names** [unverified]. The code publishes to `/arm_controller/joint_trajectory`,
   `/torso_controller/joint_trajectory`, `/gripper_controller/joint_trajectory`,
   `/head_controller/joint_trajectory` (the PAL ROS 2 names). The preflight checks they exist.
2. **Grasp** (the main risk). In Gazebo the object was held by a fixed joint, because the simulated finger friction
   would not hold it. On the robot the fingers must do it. The close command is 24.5 mm per finger for a 25 mm
   radius object, i.e. almost no squeeze. If the can slips, lower `GRIP_CLOSE_CMD` in `env.py` (e.g. 0.020) to
   squeeze harder; if the gripper trips its current limit, raise it. If the grasp height is off, change
   `GRASP_DZ`. The expert then reacts to "fingers closed on nothing" by reopening and retrying.
3. **Perception colours / bias.** Compare `/pick_place/object` with the tape measure. A constant offset can be
   subtracted in `perception.object_centre`.
4. **Table height and reach.** `--table-height` shifts the whole task vertically. A much higher/lower table needs
   a different torso height (`TORSO_HEIGHT` in `arm_control.py`) and may leave the reachable strip.
5. **Speed.** Only after clean runs: `--speed-scale 0.6 --step-scale 0.8`.

## 5. What the policy knows, and does not

* Input: hand position, object-minus-hand, pad-minus-hand (from the camera), finger opening, last grip command,
  a "holding" flag. It was trained with camera-like noise (see README), but only on the simulated noise model.
* While the hand is over the object, the camera estimate is frozen (the hand occludes it). While carrying, the
  object is assumed to sit 6 cm below the hand frame.
* Not handled: other object shapes/sizes, moving objects, clutter, a base that moves, collisions with anything
  other than the table (the arm does not avoid obstacles).

## 6. Reporting results

`run_real` writes `real_run.json` (human-judged success per episode + the perception's opinion). Record the
tape-measure perception error, grasp slips and any aborts; those are the numbers that matter for the next round
(the noise model in `perception.PerceptionNoise` should be set to the measured error, and the policy
re-trained in sim with it). Collecting data on the real robot is not implemented yet.
