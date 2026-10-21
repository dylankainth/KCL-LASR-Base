# Running pick_and_place_rl on the real TIAGo

Read this top to bottom before powering anything. **Everything here was verified in Gazebo only. Nothing has
touched the real robot.** Items marked **[unverified]** are assumptions that the preflight or the first trial
must confirm.

## 0. Safety

* Someone stands at the e-stop for every trial. The software limits (joint speed, jump check, workspace clamp,
  base speed/travel limits) are a second line of defence, not the first.
* The arm moves at 30 % speed and the hand at 40 % of the trained step by default (`--speed-scale`,
  `--step-scale`). Raise them only after clean runs.
* **The base drives itself by default** (auto-park, section 4): at most 0.12 m/s and 0.30 rad/s, 0.6 m of total
  travel, only with the arm tucked, after a confirmation. `/cmd_vel` goes through PAL's twist_mux, so the joystick
  can take over. Clear the floor around the base. There is no obstacle avoidance. `--no-park` disables driving.
* Clear the area in front of the robot. The arm unfolds from the tucked pose and passes over the table
  (the route is PAL's `home` motion reversed, then up and over).
* The first runs use the **scripted expert** (`--expert`), not the learned policy, so any failure is the
  hardware/perception, not the network.

## 1. What you need

| Item | Spec |
| :--- | :--- |
| Object | cylinder, **~5 cm diameter**, ~10 cm tall, light (<= 200 g), grippy, **saturated blue**. A tin can with a blue sleeve is fine. The policy was trained on r = 2.5 cm; a different diameter changes the finger opening the grasp logic expects (`CLOSED_ON_OBJ`, `GRIP_CLOSE_CMD` in `env.py`). |
| Pad | flat, **red**, ~14 x 14 cm (paper/card is fine), on the table |
| Table | a table with a **straight, uncovered near edge** (auto-park squares the robot to it). Table top height measured from the floor (tape measure); the sim table is 0.815 m; pass the real value with `--table-height` (the survey checks it against the depth camera and refuses a mismatch > 4 cm). |
| Layout | object and pad **roughly 10-45 cm apart sideways** (object on the robot's left, pad on its right), at similar distances from the robot. Auto-park shifts the robot so they land in the trained region (object x 0.58-0.68, y +0.05..+0.28 m; pad x 0.58-0.68, y -0.28..-0.08 m, measured from the base_footprint). Put them well inside the region, not at its edges. |
| Robot | TIAGo (PAL gripper, not the Hey5 hand) **[unverified]**, parked within ~12 cm and ~10 deg of right, facing the table 0.45-0.75 m from its edge, torso/arm tucked |

## 2. Install

On the machine that talks to the robot (same container image as the sim, or the robot's ROS 2 Humble workspace):

```bash
cd ~/ros_ws && colcon build --packages-select pick_and_place_rl && source install/setup.bash
```

Only numpy and rclpy are needed at run time (no torch).

## 3. Commissioning ladder (first time on a robot, ~30 minutes)

Do these in order. Each step either passes or tells you what to fix. Steps 1-4 are safe (no arm motion).

```bash
# 1. read-only checks: controllers, joints, URDF-vs-TF kinematics, camera TF, /cmd_vel, odometry. Moves nothing.
python3 -m pick_and_place_rl.preflight
```

Fix every `FAIL`. The most important one is `URDF FK matches TF` (< 10 mm): if it fails the IK is wrong for this
robot and nothing else matters. Warnings about `/cmd_vel` or odometry mean auto-park cannot run (use `--no-park`).

```bash
# 2. perception (check the topic names printed by the preflight; defaults are the PAL / simulation names)
python3 -m pick_and_place_rl.perception_node \
    --rgb /head_front_camera/rgb/image_raw \
    --depth /head_front_camera/depth/image_raw \
    --info /head_front_camera/rgb/camera_info
ros2 topic echo /pick_place/object   # base_footprint frame; compare with a tape measure
ros2 topic echo /pick_place/table    # [height, edge distance, edge slope, span, rms, n]
```

Perception picks the colour blob whose physical size fits (can cap 3-8 cm, pad 7-24 cm), not simply the largest, and
`run_real` refuses to move the arm when the perceived object or pad is more than 3 cm outside the trained region or
3 cm off the expected height (`env.scene_problems`): that is what a wrong `--table-height` or a wrong coloured thing
looks like. If it refuses, fix that first.

### Closed-loop mode (`--closed-loop`, off by default) **[unverified]**

The camera and the joint encoders disagree by a roughly constant amount (head tilt, table height, calibration), and no
controller that only sees that disagreement can remove it. `--closed-loop` therefore closes on things that see the hand
and the object together:

* **Grasp retry:** if the fingers close below `LOST_OPENING` with nothing attached, the gripper reopens and the object
  estimate is shifted along a spiral (8 points at 1 cm, 8 at 2 cm). The offset that finally grips is kept as a standing
  correction (max 3 cm) for the following episodes. After the spiral the episode ends. The hand slides sideways at grasp
  height between tries, so keep `--step-scale` low.
* **Touchdown release:** while carrying and descending over the pad, the object is released when the hand stays >1 cm
  above its command for 3 steps, and only within 3 cm of the expected place height and 2 cm of the pad in xy.
* **Encoder PI:** a bounded integral (max 1.5 cm) of commanded minus measured hand position, frozen during fast moves,
  contact and gripper changes, removes arm sag and lag.
* **Stale camera:** the arm holds still if no object/pad estimate has been accepted for 3 s while the hand is not what
  hides it, and aborts after 10 s.

Check in this order, arm over the object, `--approach-only` first: does the PI settle without oscillating, does a
deliberate 1.5 cm `--calib` error get found by the retry search, does touchdown fire at the pad and never in mid-air.

The camera topic names match the ones the rest of this repo uses on the real robot. Depth must be registered to
the RGB image (same size); both 16UC1 (mm) and 32FC1 (m) are handled. Tape-measure check: place the object at a
known (x, y) from the base_footprint and compare; the sim error is 0.5 cm, expect more on the real camera.
Table height (`/pick_place/table`[0]) should match your tape measure within ~1-2 cm. If the colour is not
detected, change `BLUE` / `RED` in `perception.py`. **[unverified]**

```bash
# 3. what would auto-park do? (surveys, prints the plan, does not drive)
python3 -m pick_and_place_rl.park --dry-run --table-height 0.80
# 4. camera offset (the single most valuable accuracy step, see section 5): do it at two positions
python3 -m pick_and_place_rl.calibrate --table-height 0.80 --out calib.json
# 5. approach only: stops above the object at grasp height, gripper open, never closes. Check the fingers straddle it.
python3 -m pick_and_place_rl.run_real --expert --approach-only --calib calib.json --episodes 1 --table-height 0.80 \
    --speed-scale 0.2 --step-scale 0.3
# 6. the grasp, without arm motion: finds a close command that holds the object
python3 -m pick_and_place_rl.grip_test
# 7. first full run: scripted expert, slow, one episode, with the close command from step 6
python3 -m pick_and_place_rl.run_real --expert --episodes 1 --table-height 0.80 --calib calib.json \
    --grip-close 0.0245 --speed-scale 0.2 --step-scale 0.3
# 8. the learned policy
python3 -m pick_and_place_rl.run_real --policy models/policy.npz --episodes 5 --table-height 0.80 --calib calib.json
```

After commissioning, a normal session is: park the robot roughly, place the object and pad, run step 8 (or 7).
`run_real` asks before moving, parks itself, aims the head, shows what it perceives, and asks again before every
episode. Ctrl-C holds the arm in place.

## 4. Auto-park

`run_real` surveys the scene with the arm tucked and the base still, then iterates (at most 5 corrections):

1. **Survey.** Waits until torso and head have stopped, then fits the table top (height) and its near edge from the
   depth image and takes the median object/pad estimates.
2. **Plan.** First the yaw: turn to be square to the edge (tolerance 1.5 deg). Then the shift (forward/back and
   left/right) that puts the object and pad inside the trained regions with a 2 cm margin. It refuses (and says
   why) if the layout cannot fit (spacing outside 13-56 cm, too far apart front to back), if the shift is over
   25 cm, or if it would end less than 40 cm from the table edge.
3. **Move.** TIAGo is a differential drive and cannot slide sideways, so a shift is turn, drive, turn back.
   Progress is measured from wheel odometry and every move is followed by a fresh camera survey, so odometry errors
   are corrected on the next pass.

Verified in Gazebo (section 8). `--no-park` only reports where the robot should be and stops if a move is needed.
If the table has no clean edge, `--no-park` works without it (yaw is then your job).

## 5. Why calibration matters, and how accurate the camera must be

The policy and the expert both need the object position to within ~0.5 cm. Gripper opening (about 9 cm) against a
5 cm can leaves ~1.5 cm of slack per side; a camera error beyond that along the finger axis knocks the can away.
Measured in Gazebo (scripted expert, 20 episodes each, per-axis random camera bias, one standard deviation):

| Camera bias | 0.1 cm | 0.5 cm | 1.0 cm | 2.0 cm |
| :--- | :--- | :--- | :--- | :--- |
| Success | 20/20 | 20/20 | 15/20 | 9/20 |

A constant offset of the camera estimates (hand-eye / head angle error: 1 deg of head tilt is ~1.4 cm at the table)
is removed by `calibrate.py`: the robot hovers over the perceived object with the gripper open and you nudge the
hand until the fingers straddle it. The saved offset is added to every estimate (`--calib`). Do it at two object
positions (left and right of the region); if the two offsets differ by more than ~7 mm the camera extrinsics need a
real calibration, not just a constant. Parking to a fixed spot keeps the remaining error nearly constant.

## 6. What was rehearsed (Gazebo standing in for the robot)

`python3 -m pick_and_place_rl.rehearsal --expert` runs `perception_node` + `run_real` through the same code path as
the real robot (wall clock, no teleporting, camera poses, speed limits, scripted "human" prompts). Add
`--displace X Y YAW_DEG` to start with the base displaced so the run has to park itself. It found and fixed four
problems that plain simulation had hidden (fixed sleeps vs stretched durations; a 1.5 rad single IK step; a command
trailing the speed-limited arm; the hand occluding the can so the estimate moved). **Not covered: the grasp itself**
(Gazebo's fingers cannot hold the can, and the grasp-assist joint is off in this mode), real controller behaviour,
real camera topics/colours/accuracy, real latency, the real base's low-speed behaviour.

## 7. What will most likely need tuning (in order)

1. **Controller interfaces.** The code uses the `follow_joint_trajectory` action servers of
   `/arm_controller`, `/torso_controller`, `/gripper_controller`, `/head_controller` (the interface the rest of this
   repo uses on the robot) and falls back to the `joint_trajectory` topics if an action server is missing. It prints
   which one each controller got at start-up (`[robot] command interfaces: ...`). The preflight checks both.
2. **Grasp** (the main risk). In Gazebo the object was held by a fixed joint, because the simulated finger friction
   would not hold it. On the robot the fingers must do it. Use `grip_test.py` to find the close command
   (`--grip-close`, default 0.0245 = almost no squeeze; candidates 0.0245 ... 0.0165). If the grasp height is off,
   change `GRASP_DZ`. The expert reacts to "fingers closed on nothing" or "object lost" by reopening and retrying.
3. **Base behaviour at low speed.** Gazebo's base ignores commands below ~0.12 rad/s / 0.04 m/s and its odometry
   over-reports rotation by about 2x; the driver adds a breakaway boost and the camera survey closes the loop.
   The real base is probably better, but if parking reports "the base is not moving/turning", raise `MIN_V` /
   `MIN_W` in `base.py`.
4. **Perception colours / bias.** Compare `/pick_place/object` with the tape measure; use `calibrate.py`.
5. **Table height and reach.** `--table-height` shifts the whole task vertically. A much higher/lower table may
   need a different torso height (`TORSO_HEIGHT` in `arm_control.py`).
6. **Speed.** Only after clean runs: `--speed-scale 0.6 --step-scale 0.8`.

## 8. Verified in Gazebo

| Check | Result |
| :--- | :--- |
| Table edge fit (operating head pose) | edge within 0.5 cm of the true 0.55 m; yaw within 0.06 deg; synthetic +-4 deg rotations recovered |
| Auto-park from random displacement (+-12 cm, +-10 cm, +-8 deg), 10 trials | 8/10 ended with the object and pad inside the trained region (9/10 within 1 cm); final yaw within ~1.3 deg |
| Park, then scripted-expert pick-and-place (12 separate fresh-sim runs, seeds 31-40; `results/parking/parkpnp*.json`) | 12/12 parked inside the trained region; **9/12** completed the pick-and-place. Seeds 31 and 37 each failed once and succeeded once on rerun (the sim is not deterministic); in the seed-39 failure the first grasp was aligned within 1 cm and the can still ended up off the table (simulated contact physics). With 12 runs this cannot be told apart from the ~93 % the expert reaches without parking (28/30 under camera-like noise). |
| Action-interface commands (arm, torso, gripper, head) | 3/3 expert episodes |

Results are from one simulated robot with ground-truth odometry standing in for the real wheel odometry during the
parking tests. They say nothing about the real robot.

## 9. What the policy knows, and does not

* Input: hand position, object-minus-hand, pad-minus-hand (from the camera), finger opening, last grip command,
  a "holding" flag. It was trained with camera-like noise (see README), but only on the simulated noise model.
* While the hand is over the object, the camera estimate is frozen (the hand occludes it). While carrying, the
  object is assumed to sit 6 cm below the hand frame.
* Not handled: other object shapes/sizes, moving objects, clutter, obstacles for the base or the arm (nothing
  avoids them).

## 10. Reporting results

`run_real` writes `real_run.json` (human-judged success per episode + the perception's opinion). Record the
tape-measure perception error, the calibration offsets, grasp slips and any aborts; those are the numbers that
matter for the next round (the noise model in `perception.PerceptionNoise` should be set to the measured residual
error, and the policy re-trained in sim with it). Collecting data on the real robot is not implemented yet.
