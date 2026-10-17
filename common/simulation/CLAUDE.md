# CLAUDE.md — `common/simulation`

Gazebo **Classic** + PAL `tiago_gazebo` wrapper (ROS 2 Humble). Launch files:

* `launch/sim.launch.py` — args `world_name` (default `house`, or `pick_place`), `gzclient`; includes PAL's `tiago_gazebo.launch.py` after injecting this package's `models/` into the Gazebo model path.
* `launch/nav.launch.py` — sim + Nav2 (params `config/nav2_params_scan_raw.yaml`, maps in `maps/`); `config/slam_params.yaml` for slam_toolbox mapping.
* `worlds/pick_place.world` — table (top 0.815 m), blue can (`pick_object`), red pad (`place_pad`), observer camera. Used by `tasks/pick_and_place_rl`.
* `overrides/gripper.gazebo.xacro` — PAL gripper contact params (stock `minDepth=0.1` makes contacts useless). It is **bind-mounted over** PAL's file at container start (`tasks/pick_and_place_rl/scripts/run_lasr.sh`); editing it has no effect unless it is mounted.

Editing worlds/models/launch needs a rebuild: `colcon build --packages-select simulation gazebo_grasp_attach` (no `--symlink-install`).
Gazebo startup is flaky: the arm tuck can fail and stale `gzserver`/`play_motion2` processes break reruns (see `tasks/pick_and_place_rl/scripts/job.sh` for the cleanup/retry loop).
