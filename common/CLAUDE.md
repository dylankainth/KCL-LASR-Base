# CLAUDE.md — `common/`

Infrastructure packages (nodes, services, interfaces, sim). Tasks and skills call these over ROS topics/services/actions.

| Folder | Packages | Role |
| :--- | :--- | :--- |
| `vision/` | `lasr_vision_interfaces` (msg/srv/action), `lasr_vision_yolo` (`/yolo/detect`, `/yolo/detect3d`, `/yolo/detect_pose`, `/yolo/detect3d_pose`), `lasr_vision_open_vocabulary` (Grounding DINO / YOLO-World + SAM, `open_vocab/detect`), `lasr_vision_reid`, `lasr_vision_eye_tracker`, `lasr_vlm` | Perception services. README lists topics. In sim, disable the depth `/ 1000.0` conversion in the yolo service node. |
| `speech/` | `lasr_speech_recognition_interfaces`, `lasr_speech_recognition_whisper` | Whisper STT (`transcribe_microphone_server`) |
| `language/` | `lasr_llm`, `lasr_llm_interfaces` | LLM inference via Ollama; task-specific services under `lasr_llm/nodes/` |
| `manipulation/` | `lasr_manipulation_actions` | Grasp planning/arm actions for TIAGo (python dir is `manipulation_actions/`, not matching the package name) |
| `navigation/` | (placeholder, no packages) | Nav2 is launched from `simulation/launch/nav.launch.py` |
| `simulation/` | `simulation` | Gazebo Classic worlds (`house.world`, `pick_place.world`), launch, maps, nav2/slam params, `overrides/gripper.gazebo.xacro` (see its own CLAUDE.md) |
| `interfaces/` | `robot_ui` | Tablet web UI (Next.js) over rosbridge; used for start/confirm in HRI, GPSR, restaurant |
| `helpers/` | `cv2_img`, `lasr_helper_interfaces` | Image conversion helpers; flagged as possibly abandoned |
| `gazebo_grasp_attach/` | C++ Gazebo plugin | Fixed joint hand↔object, toggled by a topic (sim grasp assist) |

Rules of thumb
* A new message/service goes in the matching `*_interfaces` package (CMake/`rosidl`), not in the consumer.
* Services that load models (YOLO, Whisper, VLM) should load once at node start and keep weights out of git; per-package `requirements.txt` + ament_virtualenv.
* READMEs here are short; when you add a node, document its topics/services in the folder README.
