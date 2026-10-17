# CLAUDE.md — LASR Base (ROS 2 Humble, TIAGo, RoboCup@Home)

Monorepo of King's College London LASR@Home. A colcon workspace of ROS 2 Python packages for a PAL TIAGo robot.
Default branch: `ros2` (`ros1` is the legacy Noetic code). Origin is the user's fork; `upstream` is LASR-at-Home/Base.

## Layout (three layers, dependencies point downwards)

| Layer | Path | What lives there |
| :--- | :--- | :--- |
| Tasks | `tasks/<Task>/` | Full RoboCup tasks (GPSR, HRI, restaurant, pick_and_place) as YASMIN state machines, plus `pick_and_place_rl` (a separate imitation-learning project, has its own `CLAUDE.md`) |
| Skills | `skills/` | One ROS package `skills`, python module `lasr_skills`: reusable YASMIN `State`s (`Say`, `GoToLocation`, `Detect3D`, `Listen`, ...) |
| Common | `common/` | Infrastructure: `vision`, `speech`, `language` (LLM via Ollama), `manipulation`, `simulation` (Gazebo), `interfaces` (tablet UI), `helpers`, `gazebo_grasp_attach` (C++) |
| Docs | `documentation/` | WSL2+Apptainer setup, YASMIN porting guide, ament_virtualenv guide |

Each layer has its own `CLAUDE.md` with details. Read the nearest one before editing.

## Environment (important)

* Code only runs inside the **Apptainer container** built from `ros2.def` (needs PAL's non-public `tiago_humble_os.sif`).
  Workspace in the container is `/opt/robocup_ws`; on the author's WSL setup it is `~/ros_ws/src/Base`.
* The user develops on **Windows** (this checkout) but builds/runs in **WSL**. Windows ↔ WSL sync is `rsync` via
  `tasks/pick_and_place_rl/scripts/sync.sh`. You cannot run ROS, Gazebo or `colcon` from the Windows shell, so
  **you usually cannot execute or test ROS code here**. Say so rather than claiming something works.
  Pure-numpy modules (e.g. `pick_and_place_rl/kinematics.py`, `policy.py`, `perception.py`) can be reasoned about or run with plain Python.
* Build: `colcon build --packages-select <pkg>` then `source install/setup.bash`.
  **Do not use `--symlink-install`** for packages with pip deps (`ament_virtualenv` hooks `install`, not `develop`;
  see `documentation/ament_virtualenv_guide.md`).
* Sim: `ros2 launch simulation sim.launch.py` (house) or `... world_name:=pick_place gzclient:=False`; navigation: `ros2 launch simulation nav.launch.py`.

## Conventions

* **Python formatting: `black` 24.3.0** (CI `--check` on every PR, `.github/workflows/qc`). Run `black <files>` before committing.
* **State machines use YASMIN, not SMACH.** `yasmin.State` / `StateMachine`, outcomes strings, the **blackboard** (`blackboard["key"]`) in place of userdata,
  `add_input_key` / `add_output_key`, `yasmin.YASMIN_LOG_INFO(...)` for logs. Rules (from `documentation/README.md`):
  `set_sigint_handler(True)` / `handle_sigint=True` only on the top-level SM; use dedicated service/action/subscriber states
  (`yasmin_ros.ServiceState/ActionState/...`); **never call `rclpy.spin()`** in states (YASMIN/executor threads own spinning).
  Porting from SMACH: `documentation/YASMIN_PORT.md`.
* Common state outcomes: `succeeded`, `failed` (action states also `aborted`, `canceled`). Keep this vocabulary.
* Package types: `ament_python` everywhere except `common/gazebo_grasp_attach` (C++/CMake). Entry points go in `setup.py` `console_scripts`; run with `ros2 run <pkg> <script>`.
* Packages needing pip deps ship `requirements.txt`, list `<build_depend>ament_virtualenv</build_depend>` and `<pip_requirements>requirements.txt</pip_requirements>`
  in `package.xml`, and use the `InstallCommand` + `ament_virtualenv.install.install_venv(..., source_dir=_here)` boilerplate in `setup.py`
  (copy from `skills/setup.py`). Also add `requirements.txt` to `data_files`.
* New package checklist: `package.xml`, `setup.py`, `setup.cfg`, `resource/<pkg_name>` marker file, python package dir, README.
* Task parameters (locations, objects, etc.) live in `tasks/<Task>/config/*.yaml` and are passed with `--params-file`.

## Gotchas

* Several `setup.py` entry points reference modules that do not exist (e.g. `skills` lists `set_initial_pose`; `restaurant` lists `detect_wave`; `lasr_manipulation_actions` has python dir `manipulation_actions` but entry points say `lasr_manipulation_actions.*`). Verify before relying on, or fixing, an entry point.
* Skills file names are snake_case but class names are CamelCase and re-exported from `lasr_skills/__init__.py`; add new skills there.
* Heavy models (YOLO, Whisper, VLM, Ollama) need GPU memory; don't assume they all run at once.
* `tasks/GPSR/external/` is vendored third-party; don't reformat it.
* Many stale feature branches on `upstream`; the working branches here are `ros2` and `pick-and-place-rl`.

## Working style for this repo

* Match surrounding code; don't reformat unrelated files (black diff noise).
* Prefer extending an existing skill/state over writing a one-off in a task.
* Update the package README when behaviour or run commands change.
* Commits: short imperative subject; the user commits only when asked.
