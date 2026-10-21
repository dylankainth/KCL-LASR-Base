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
| Camera perception (colour + depth), table height/edge | `perception.py`, `perception_node.py`, `table.py` |
| Offline perception stress test, regression tests | `synthetic_check.py`, `test/test_offline.py` |
| Parking: base driver, planner, Gazebo test | `base.py`, `park.py`, `park_test.py` |
| Real-robot runner, preflight, calibration, grip test, rehearsal | `run_real.py`, `preflight.py`, `calibrate.py`, `grip_test.py`, `rehearsal.py`, `REAL_ROBOT.md` |

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

## Results (Gazebo, 30 unseen random tasks, seed 777)

| Policy | Training data | Success | Mean place error |
| :--- | :--- | :--- | :--- |
| Scripted expert | - | 10 / 10 | - |
| `eval_bc_only.json` | 120 expert episodes (70 % with action noise, DART), 17.8 k samples | 19 / 20 | 0.1 cm |
| `models/policy.npz` (`eval_bc_dagger.json`) | + 60 DAgger episodes (6.5 k samples) | **30 / 30** | 0.1 cm |

The 5-episode demo video (different seed) also succeeded 5/5. The policy is a 3x256 MLP (state in, 4-D action out)
trained by behaviour cloning on the expert's labels; DAgger rounds relabel the states the learned policy visits.
No reinforcement learning was used: with a privileged-state expert and Gazebo running at ~1x real time, imitation
(BC + DART + DAgger) needs ~10^4 samples, where RL would need orders of magnitude more simulator time.

### With camera perception and noise (Gazebo)

| Test | Policy | Success |
| :--- | :--- | :--- |
| Poses from the head-camera pipeline (`perception_node`), seed 777 | `policy.npz` | 19 / 20 (1.0 cm mean place error) |
| Same, 30 episodes | `policy_noise_dagger.npz` | 24 / 30 |
| Camera-like noise on the observation (1 cm bias, 4 mm jitter, stale frames), seed 778 | `policy.npz` | 25 / 30 |
| Same | `policy_noise_dagger.npz` | 21 / 30 |
| Same | scripted expert | 24 / 30 |

Perception alone (`perception_check`): object 0.55 cm, pad 0.16 cm mean error over 20 random placements, no misses.
That 0.55 cm was a systematic offset of the old object estimator (see "Offline perception stress test" below); after the
fix the same check gives object **0.05 cm** (max 0.07), pad 0.16 cm, no misses. The 19/20, 24/30 and 25/30 rows above
were measured with the old estimator and have not been re-run.

### Offline perception stress test (no ROS, numpy only)

`python3 -m pick_and_place_rl.synthetic_check` renders the table, can and pad with a ray-cast pinhole camera and
degrades it like a real depth camera (range-dependent depth noise, mm quantisation, dropouts, flying pixels at
edges, RGB/depth misregistration, dim/bright/coloured light, glare, distractors, head-tilt error). The noise figures
are typical Xtion/Astra-class **assumptions, not measurements of the TIAGo camera**. It found three things, now fixed:

| Finding | Before | After |
| :--- | :--- | :--- |
| Object estimate bias on perfect data (the cap band included a strip of the can's side, pulling the median toward the camera) | 0.85 cm | 0.23 cm (Gazebo: 0.55 → 0.05 cm) |
| A bigger blue blob in view (folder, chair) was taken for the can | 43 cm error | ignored; blobs are chosen by physical size |
| Pad lost in a dim room (value threshold 0.25) | missed | found (threshold 0.14) |

Remaining sensitivity (these are systematic, `calibrate.py` removes a constant offset): 1 deg head-tilt error gives
~1.2 cm table-height error and ~0.8 cm object error; 4 px depth/RGB misregistration ~0.6 cm. The table-edge distance
has ~0.9 cm stride-quantisation bias (depth subsampled by 3), harmless for parking (2 cm margin).
`python3 -m unittest discover -s test` runs the offline regression tests.

### Closed-loop mode

`run_real --closed-loop` adds contact and encoder feedback over the policy (`closed_loop.py`, see `REAL_ROBOT.md`). The
unit tests cover the controllers and the `env` hooks against a fake scene; none of it has run on Gazebo or hardware.
The extra DAgger round with noisy observations (`policy_noise_dagger.npz`) did **not** help, so `policy.npz` stays
the default. The expert is no better than the policy under this noise, so the noise level, not the network, is
what limits success; the numbers are within the spread expected from 30 episodes. Results files are in `results/`.

### Why the policy is not better, and what limits success

* **Not too few epochs: overfitting.** Training loss fell from 0.17 to 0.0045 while held-out loss rose from 0.18
  (epoch 25) to 0.27 (epoch 375). `scripts/train_bc.py` now keeps the best held-out epoch (early stopping) and
  supports dropout, weight decay and input noise. A retrained policy (dropout 0.1, weight decay 1e-3, best epoch 20)
  scored 28/30 vs 29/30 for `policy.npz` and 28/30 for the expert on the same 30 noisy episodes (seed 901,
  0.5 cm bias; `results/cmp_*.json`): no measurable gain, so `policy.npz` stays.
* **Camera accuracy decides the outcome.** Scripted expert, 20 episodes per point, per-axis random camera bias
  (`results/sweep/`):

  | Bias (1 sigma) | 0.1 cm | 0.5 cm | 1.0 cm | 2.0 cm |
  | :--- | :--- | :--- | :--- | :--- |
  | Success | 20/20 | 20/20 | 15/20 | 9/20 |

  The gripper opens ~9 cm for a 5 cm can, leaving ~1.5 cm per side; beyond that the finger knocks the can away. So
  the real-robot plan is to measure and remove the camera offset by hand (`calibrate.py`), not to train longer.
* Policies imitate the expert, so more demonstrations can make them more consistent but cannot beat the expert.

### Parking (Gazebo)

`park.py` squares the robot to the table edge (depth camera) and shifts it so the object and pad land in the trained
region. From random displacements (+-12 cm, +-10 cm, +-8 deg): 8 of 10 trials ended inside the trained region
(9 of 10 within 1 cm), final yaw within about 1.3 deg (`results/parking/`). Parking followed by the scripted expert's
pick-and-place (12 fresh-sim runs): 12/12 parked in the region, 9/12 completed the pick-and-place (see
`REAL_ROBOT.md` section 8 for the failures). The simulated base slips at low speed and its odometry over-reports
rotation, so these tests use Gazebo's ground-truth odometry as a stand-in for the real robot's.

The real-robot path is described in `REAL_ROBOT.md` (nothing here has run on the real TIAGo).

Caveats: ground-truth poses (no perception), a grasp-assist joint (see above), fixed object shape, and
evaluation on the same table/pad regions used for training. Run `scripts/run_lasr.sh` etc. from the WSL host
(they assume the layout in `documentation/WSL_APPTAINER_SETUP.md`).
