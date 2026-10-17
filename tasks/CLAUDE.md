# CLAUDE.md — `tasks/`

One ROS package per RoboCup@Home task. Each is a YASMIN `StateMachine` composed mostly from `lasr_skills` plus task-specific states.

| Task | Package / module | Entry | Notes |
| :--- | :--- | :--- | :--- |
| `GPSR` | `GPSR` | `ros2 run GPSR sm --ros-args --params-file tasks/GPSR/config/params.yaml` | LLM (Ollama, `gemma3`) plans skills from a spoken/typed command (`planner.py`, `prompts.py`); `dispatch_skill.py` executes; `world.py` + `config/{locations,objects,people,skills,general_knowledge}.yaml` describe the world; `params.yaml`: `simulation`, `input_mode: keyboard|...`. `external/` is vendored. Has its own README. |
| `HRI` | `HRI` | `sm`, `start_sm`, `seat_guest`, `recognise`, `place_bag`, `n_and_d` | Receptionist: greet, name/drink, face recognition, seating. Arena yamls in `config/`. Uses `robot_ui` for start button. |
| `restaurant` | `restaurant` | `sm`, `survey`, `detect_waving_person_rgb` | Find waving customer, take order. `config/lab.yaml`. |
| `pick_and_place` | `pick_and_place` | `state_machine`, `test_detect` | RoboCup pick-and-place where a *human* does the manipulation on voice instruction; robot perceives and speaks. |
| `pick_and_place_rl` | `pick_and_place_rl` | python modules (`python3 -m ...`) | Unrelated to the above: learned arm policy, sim + real TIAGo. **Read its own `CLAUDE.md`.** Not a YASMIN SM. |

## Task package layout (HRI/GPSR/restaurant/pick_and_place)

```
tasks/X/  package.xml  setup.py  setup.cfg  resource/X  requirements.txt?
          X/state_machine.py   # top-level SM + main()
          X/states/*.py        # task-specific states, exported in states/__init__.py
          config/*.yaml        # params, locations; launch/*.launch.py
```

Top-level `main()` pattern (see `documentation/YASMIN_PORT.md`): `rclpy.init()` → `yasmin_ros.set_ros_loggers(node)` → build SM
(`handle_sigint=True` only here) → optional `YasminViewerPub` → `sm()` → `rclpy.shutdown()`. A helper node spun on an
`EventsExecutor`/`MultiThreadedExecutor` in a thread provides parameters. Transitions map every outcome of every state; unmapped outcomes raise at build time.

When adding a task: copy the smallest existing one (`restaurant` or `pick_and_place`), register the entry point, add config yaml and README.
