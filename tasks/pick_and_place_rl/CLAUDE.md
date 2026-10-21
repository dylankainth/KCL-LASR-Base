# CLAUDE.md — `tasks/pick_and_place_rl`

Imitation-learned (BC + DART + DAgger; **no RL**) TIAGo tabletop pick-and-place: pick a blue 5×10 cm cylinder, place on a red pad.
Developed in Gazebo; the real-robot path (`run_real`) is **written but has never touched hardware**. Keep that distinction in every claim you make.
User-facing docs: `README.md` (design + results), `REAL_ROBOT.md` (commissioning runbook), `LAPTOP_SETUP.md` (getting the uni laptop's WSL connected to the real robot; read it if asked to set up or connect to TIAGo). Read them before changing behaviour.

## Not a YASMIN task
Plain python modules run as `python3 -m pick_and_place_rl.<module>` (no `console_scripts`). Package is `ament_python`, deps: rclpy, gazebo_msgs, control_msgs, trajectory_msgs, `simulation`.
Runtime needs only numpy + rclpy (torch is host-only, for training).

## Module map (`pick_and_place_rl/`)

| Layer | Modules |
| :--- | :--- |
| Math | `kinematics.py` (URDF FK/IK, damped least squares, numpy; `Chain`, `solve_ik`, `rotation_from_axes`) |
| ROS I/O | `robot.py` `TiagoScene` (node, joint/model/perception subscriptions, trajectory output via action with topic fallback, speed limiting, `MAX_JUMP`), `arm_control.py` `ArmController` (Cartesian hand moves, routes `go_ready`/hover, `Q_SEED`, `TORSO_HEIGHT`, `HOVER_Z`), `head.py`, `base.py` (`/cmd_vel` driver with limits) |
| Task | `env.py` `PickPlaceEnv` + `expert_action(obs)` (12-D obs → 4-D action; regions, constants), `policy.py` `MLPPolicy` (numpy) |
| Learning | `collect.py` (BC/DART/DAgger data), `scripts/train_bc.py` (host, torch → `.npz`), `evaluate.py`, `video.py` |
| Perception | `perception.py` (HSV colour + depth → base_footprint points; pure numpy), `perception_node.py` (publishes `/pick_place/object`, `/pad`, `/table`), `table.py` (table height + near edge fit), `perception_check.py`, `snapshot.py` |
| Real robot | `run_real.py` (main runner), `park.py` (auto-park), `preflight.py` (read-only checks), `calibrate.py` (camera offset by eye), `grip_test.py`, `rehearsal.py` (real code path vs Gazebo), `selftest.py` (scripted-human test of calibrate/grip_test), `park_test.py`, `base_probe.py`, `topic_probe.py` |
| Closed loop | `closed_loop.py` (pure numpy: `HandTracker` encoder PI, `FingerSettle`, `GraspSearch` retry spiral + learnt object bias, `touchdown`); wired into `env.step` / `_reperceive` behind `env.closed_loop` / `run_real --closed-loop` (needs `--grip-close` < 0.020), **unverified on hardware**. `closed_loop_test.py` is its Gazebo harness (real-robot code path, fault injection into perception, **emulated** finger contact and hard surface; scenarios `track stale carry touch_emul learn flow`) |
| Offline tests | `synthetic_check.py` (ray-cast imperfect camera vs perception/table), `test/test_offline.py` (`python3 -m unittest discover -s test`, numpy only, runs in WSL host python or any env with numpy) |
| Scripts | `scripts/{sync,start_sim,run_lasr,job}.sh` — WSL host helpers (see below) |
| Artifacts | `models/policy.npz` (default), `policy_noise_dagger.npz`; `results/*.json` evaluation outputs |

## Contracts that must stay in sync
* **Observation (12)**: hand xyz, object−hand, pad−hand, finger opening, gripper cmd, attached flag. **Action (4)**: dx,dy,dz (≤ `MAX_STEP`=0.012 m per `DT`=0.1 s), gripper (>0.5 closes). Changing either invalidates `models/*.npz` and all datasets; the policy must be retrained.
* `MAX_STEP` is duplicated in `scripts/train_bc.py`; `GRASP_DZ` in `env.py` and `robot.py`. Change both.
* Trained regions `OBJ_REGION`, `PAD_REGION`, `HAND_START_REGION`, `WORKSPACE_LO/HI` in `env.py` are mirrored by auto-park (`park.py`) and documented in `REAL_ROBOT.md`.
* Sim table top is 0.815 m; `--table-height` applies a z-offset to the whole task.
* Kinematics assume base frame == world frame in sim (env re-pins the base). On the robot, everything is in `base_footprint`.
* Frames: all perception outputs are `PointStamped` in `base_footprint`; the gripper axes are approach = local +x, finger opening = local −y (`robot.py`).

## Sim honesty (do not paper over)
Gazebo Classic fingers cannot hold the can; in sim the `gazebo_grasp_attach` plugin adds a fixed joint when fingers close (`attach_object`/`detach_object`). Arm is position-driven with unlimited force and ignores collisions, so routes are hand-planned to clear the table. The simulated base slips at low speed and over-reports rotation. `rehearsal` mode (`sim=False`) turns the grasp assist off.

## Safety rules for real-robot code
Any change touching `robot.py`, `arm_control.py`, `base.py`, `run_real.py` must keep: speed/jump limits, workspace clamp, confirmation before motion, zero command on exit, the e-stop assumption, base moves only with arm tucked, and low default `--speed-scale`/`--step-scale`. Never loosen defaults without the user asking. Items tagged **[unverified]** in the runbook stay that way until hardware tests.

## Running (WSL host, Apptainer; you cannot do this from Windows)
`scripts/sync.sh` rsyncs the Windows checkout into `~/ros_ws/src/Base`. `scripts/job.sh LOG module args...` starts a fresh sim, waits for "Arm tucked", runs one module inside the container, tears down, appends `JOB FINISHED` to LOG. Never run two controllers at once; stale `gzserver`/`play_motion2` break runs. Sim: `ros2 launch simulation sim.launch.py world_name:=pick_place gzclient:=False`. Commands in `README.md`/`REAL_ROBOT.md`.

## Workflow tips
* Verify numerically-sensitive changes with the pure-numpy parts first (`kinematics`, `table`, `perception`, `policy`, `expert_action`), then in Gazebo via `evaluate.py` with a fixed seed (baseline: seed 777, `policy.npz` 30/30 on ground truth).
* Record new evaluation outputs in `results/` and update the README tables; don't claim improvement from <30 episodes without noting variance.
* Code style: compact, module docstring with usage first, constants at the top with a trailing comment explaining units. This package is currently **not black-formatted** (~360 lines over 88 chars; CI's black check would fail on a PR to `ros2`). Match the existing style in files you touch; only run `black` on a file as a deliberate, separate formatting commit.
