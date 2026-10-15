# pick_and_place_rl

TIAGo picks a can off a table and places it on a target pad in Gazebo, using a policy
**learned from demonstrations** (behaviour cloning with DART noise, then DAgger).

This is separate from `tasks/pick_and_place` (the RoboCup task, where a human does the manipulation).

## What is here

| Piece | Where |
| :--- | :--- |
| Scene: table, can, red target pad, observer camera | `common/simulation/worlds/pick_place.world` |
| Grasp assist (fixed joint hand ↔ can, on a topic) | `common/gazebo_grasp_attach` |
| Gripper contact parameters override | `common/simulation/overrides/gripper.gazebo.xacro` |
| URDF forward/inverse kinematics (numpy) | `pick_and_place_rl/kinematics.py` |
| ROS interface, Cartesian arm control | `robot.py`, `arm_control.py` |
| Environment + scripted expert | `env.py` |
| Data collection (BC / DART / DAgger) | `collect.py` |
| Training (PyTorch, host) | `scripts/train_bc.py` |
| Policy inference (numpy, container) | `policy.py` |
| Evaluation + video | `evaluate.py`, `video.py` |

## Task, observation, action

* Object: 5 cm × 10 cm cylinder at a random pose on the table; target: a red pad at a random pose.
* Observation (12): hand xyz, object − hand, pad − hand, finger opening, gripper command, attached flag.
  Object and pad poses are Gazebo ground truth (perception is bypassed).
* Action (4): hand displacement `dx, dy, dz` (≤ 1.2 cm per 0.1 s) and a gripper open/close command.
  Joint targets come from damped-least-squares IK that keeps the gripper pointing straight down.

## Honest notes on the simulation

* **Grasp assist.** Gazebo Classic's ODE could not hold a small object between the simulated fingers
  (the object is ejected or slips; see the debugging notes in the commit history). When the fingers have
  closed on the object, `gazebo_grasp_attach` adds a fixed joint between the hand and the object, and removes
  it when the gripper opens. The arm, gripper, contacts, release and placement are all still simulated.
* **Gripper contacts.** The stock PAL finger settings (`minDepth = 0.1 m`) make contacts useless, so
  `overrides/gripper.gazebo.xacro` is bind-mounted over them (done by `run_lasr.sh`, see the setup guide).
* **Base pinned by checking.** The kinematics assume the base frame is the world frame. The env checks the
  base pose every step and re-pins it if it drifted.
* The arm is position-driven with unlimited force, so it ignores static collisions (e.g. the table). The
  routes (`ArmController.go_ready`, hover height) are therefore planned to stay clear of the table.

## Run (inside the Apptainer container, see `documentation/WSL_APPTAINER_SETUP.md`)

```bash
# terminal 1: scene
ros2 launch simulation sim.launch.py world_name:=pick_place gzclient:=False

# terminal 2: collect expert demos with DART noise
python3 -m pick_and_place_rl.collect --episodes 120 --noise 0.008 --out data/bc_round0.npz
# host: train
python scripts/train_bc.py --data data/bc_round0.npz --out data/policy0.npz
# DAgger: the learned policy drives, the expert labels
python3 -m pick_and_place_rl.collect --episodes 60 --policy data/policy0.npz --beta 0.3 --out data/dagger1.npz
# evaluate + record a video
python3 -m pick_and_place_rl.evaluate --policy data/policy1.npz --episodes 30 --video out.mp4
```
