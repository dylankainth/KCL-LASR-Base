# CLAUDE.md — `skills/` (package `skills`, module `lasr_skills`)

Reusable YASMIN states / small state machines that tasks compose. Source is in `src/lasr_skills/` (`package_dir={"": "src"}`).
Import in tasks as `from lasr_skills import Say, GoToLocation, Detect3D, ...`.

## Current skills (all re-exported in `src/lasr_skills/__init__.py`)

| Area | Skills |
| :--- | :--- |
| Speech | `Say` (action `/tts_engine/tts`, falls back to logging if `tts_msgs` missing), `Listen`, `AskAndListen` |
| Navigation | `GoToLocation` (Nav2 `BasicNavigator`), `SafeGoToLocation`, `ContinuousGoToLocation`, `Rotate`, `FollowPerson` |
| Perception | `Detect3D`, `Detect3DInArea`, `DetectAllInPolygon`, `DetectKeypoints3D`, `DetectWave`, `DetectDoorOpening`, `DescribePeople`, `WaitForPersonInArea`, `StartEyeTracker/StopEyeTracker`, `vision/{get_image,crop_image_3d}` |
| Motion / HRI | `PlayMotion` (PAL play_motion2, `config/motions.yaml`), `LookToPoint`, `HandoverObject`, `ReceiveObject` |
| Misc | `Wait`, `StartDoorSM` (`start_task.py`), `safety.py` (node, `config/safety.yaml`, `launch/safety.launch.py`) |

## Writing a skill

```python
import yasmin
from yasmin import State

class MySkill(State):
    def __init__(self, some_arg=None):
        super().__init__(outcomes=["succeeded", "failed"])
        self.add_input_key("thing")        # blackboard inputs
        self.add_output_key("result")      # blackboard outputs
    def execute(self, blackboard):
        yasmin.YASMIN_LOG_INFO("...")
        blackboard["result"] = ...
        return "succeeded"
```

* Prefer `yasmin_ros.ServiceState` / `ActionState` / `PublisherState` / `MonitorState` for ROS calls over hand-rolled clients (threading is handled for you). See `say.py` for an `ActionState` with `create_goal_handler`.
* Get the shared node with `yasmin_ros.logger_node` (set up by `yasmin_ros.set_ros_loggers()` in `main`). Parameters (e.g. `GoToLocation(location_param="start_pose")` reads `start_pose.position.x`...) come from the task's yaml.
* Allow a value either as constructor arg or blackboard key (pattern in `GoToLocation`): constructor wins, else blackboard.
* Optional deps (like `tts_msgs`) are imported in `try/except ImportError` so the package imports on dev machines.
* Add the class to `__init__.py`; add a `console_scripts` entry in `setup.py` only if the skill has a runnable `main()` (a standalone `main` is how a skill is tested: `ros2 run skills say`). Check that the module named in an entry point exists: `set_initial_pose` currently does not.
* New pip dependency → `requirements.txt` (built into a virtualenv by `ament_virtualenv`; do not build with `--symlink-install`).
* Services skills call live in `common/` (`/yolo/detect3d`, `/open_vocab/detect`, whisper action, `lasr_llm`); interface definitions in `common/vision/lasr_vision_interfaces` etc.
