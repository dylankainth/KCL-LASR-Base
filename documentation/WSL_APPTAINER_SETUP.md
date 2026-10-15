# Running LASR Base on Windows (WSL2 + Apptainer + NVIDIA GPU)

This guide gets the TIAGo simulation (Gazebo, ROS 2 Humble) running from this repo on a Windows PC with an NVIDIA GPU, using WSL2 and Apptainer.

> **Status:** the sections marked ✅ were verified on Windows 11 + WSL2 (Ubuntu 24.04) + RTX 3060 12 GB, Apptainer 1.5.3. Sim launch was verified by node/topic listing (Gazebo server+client, controllers, sensors, MoveIt); the Gazebo window was not visually checked. Steps marked ⚠️ are the likely trouble spots.

---

## 0. What you need

| Requirement | Notes |
| :--- | :--- |
| Windows 11 (or Windows 10 22H2 with current WSL) | WSLg provides GUI windows (Gazebo, RViz) |
| NVIDIA GPU + current **Windows** driver | Do **not** install a Linux GPU driver inside WSL |
| ~60 GB free disk, 16 GB+ RAM | Image build installs ROS, Ollama, Node, builds the workspace |
| `tiago_humble_os.sif` | PAL's TIAGo Humble base image. Not public: ask a LASR team member / Dr. Gerard Canal |

Laptop 3060s have 6 GB VRAM, desktop 3060s have 12 GB. Running the sim plus every model (YOLO, VLM, Whisper, Ollama LLM) at once will not fit in 6 GB. Run only what your task needs.

---

## 1. WSL2 setup

In PowerShell (admin):

```powershell
wsl --install -d Ubuntu-22.04     # or Ubuntu-24.04
wsl --update
wsl -l -v                         # VERSION must be 2
```

Optional: cap WSL memory so Windows stays usable. Create `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=16GB
swap=8GB
```

Then `wsl --shutdown` and reopen Ubuntu.

Check the GPU is visible inside WSL:

```bash
nvidia-smi
```

If this fails, update the Windows NVIDIA driver and run `wsl --update`. Do not continue until it works.

**Important:** keep the repo and the `.sif` files on the Linux filesystem (`~/...`), **not** under `/mnt/c/...`. Builds on the Windows mount are very slow and can break permissions.

---

## 2. Install Apptainer in WSL

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
apptainer --version
```

If the PPA does not support your Ubuntu release, download the `.deb` from the Apptainer GitHub releases page and `sudo apt install ./apptainer_*.deb`.

---

## 3. Get the repo and base image

```bash
mkdir -p ~/ros_ws/src && cd ~/ros_ws/src
git clone https://github.com/LASR-at-Home/Base.git     # or your fork / this repo
cd Base
```

Copy `tiago_humble_os.sif` into the repo root (`~/ros_ws/src/Base/`).

Edit the `From:` line in `ros2.def`. It is hard-coded to another user's path:

```
From: /home/rexy/tiago_humble_os.sif
```

Change it to your path, e.g. `From: /home/<you>/ros_ws/src/Base/tiago_humble_os.sif`. Do not commit this change.

---

## 4. Build the container image

From the repo root:

```bash
apptainer build ros2.sif ros2.def
```

This installs ROS packages, `ament_virtualenv`, `ros2_numpy`, `yasmin`, Ollama, rosbridge and Node 18, then builds the robot UI. It takes a long time and needs network access.

⚠️ If it fails with temp-space errors, point Apptainer at a bigger temp dir: `export APPTAINER_TMPDIR=$HOME/apptainer_tmp && mkdir -p $APPTAINER_TMPDIR`.

⚠️ If rootless build is refused, try `sudo apptainer build ros2.sif ros2.def`.

---

## 5. Run the container with GPU + GUI

```bash
apptainer run --nv -B /run/user/$UID ros2.sif
```

If `nvidia-smi` does not work inside the container (WSL keeps its NVIDIA libs in `/usr/lib/wsl/lib`):

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
apptainer run --nv -B /usr/lib/wsl -B /run/user/$UID ros2.sif
```

If GUI windows do not appear, check inside WSL that `echo $DISPLAY $WAYLAND_DISPLAY` are set (WSLg does this), then try binding the sockets:

```bash
apptainer run --nv -B /usr/lib/wsl -B /tmp/.X11-unix -B /mnt/wslg -B /run/user/$UID ros2.sif
```

Inside the container, verify:

```bash
nvidia-smi
glxinfo -B | grep -i renderer     # want your NVIDIA GPU or D3D12; "llvmpipe" means software rendering (slow)
```

---

## 6. Build the workspace

The image builds the dependencies it clones, **not this repo**. Build this repo from the container (workspace root is the parent of `src/`):

```bash
source /opt/env.sh                 # sets up ROS + the image's workspace
cd ~/ros_ws
colcon build
source install/setup.bash
```

Some packages have extra Python requirements (`requirements.txt`, handled via `ament_virtualenv`). See `documentation/ament_virtualenv_guide.md`. Build one package at a time if the full build is too heavy:

```bash
colcon build --packages-select simulation lasr_skills
```

---

## 7. Launch the simulation

```bash
ros2 launch simulation sim.launch.py      # TIAGo in Gazebo, "house" world, MoveIt, arm tucked
ros2 launch simulation nav.launch.py      # use this instead if you are using YOLO
```

Mapping and navigation helpers (from `common/simulation/README.md`):

```bash
ros2 run slam_toolbox async_slam_toolbox_node --ros-args --params-file <path>/slam_params.yaml
ros2 run nav2_map_server map_saver_cli -f <path_map>
```

`slam_toolbox` may not be installed in the image.

⚠️ If Gazebo is unusably slow or crashes at start, it is probably on software rendering (see the `glxinfo` check) or a GPU/driver mismatch. Fallback: run `gzserver` headless and view the robot in RViz.

---

## 8. Adding the rest of the stack (one piece at a time)

Add these only after the sim is stable, and watch VRAM with `nvidia-smi`.

| Piece | Package | Notes |
| :--- | :--- | :--- |
| YOLO detection | `common/vision/lasr_vision_yolo` | services `/yolo/detect`, `/yolo/detect3d`, pose variants |
| Other vision | `lasr_vision_open_vocabulary`, `lasr_vision_reid`, `lasr_vlm`, `lasr_vision_eye_tracker` | VLM and open-vocab are the heaviest |
| Speech | `common/speech/lasr_speech_recognition_whisper` | Whisper STT |
| LLM | `common/language/lasr_llm` | needs Ollama running (installed in the image) |
| Manipulation | `common/manipulation/lasr_manipulation_actions` | `manipulation.launch.py` |
| Robot UI | `common/interfaces/robot_ui` | rosbridge + Next.js; `scripts/dev` for development |
| Tasks | `tasks/{HRI,GPSR,restaurant,pick_and_place}` | built on `skills/` (`lasr_skills`) |

Each package has its own README with launch details.

---

## 9. Troubleshooting checklist

| Symptom | Try |
| :--- | :--- |
| `nvidia-smi` fails in WSL | Update Windows NVIDIA driver, `wsl --update`, `wsl --shutdown` |
| `nvidia-smi` fails only in container | Bind `/usr/lib/wsl`, set `LD_LIBRARY_PATH=/usr/lib/wsl/lib` |
| No GUI windows | Check `$DISPLAY`/`$WAYLAND_DISPLAY`, bind `/tmp/.X11-unix` and `/mnt/wslg` |
| Gazebo very slow | `glxinfo` shows `llvmpipe`: fix GPU rendering, or run headless |
| Build is very slow / permission errors | Move everything off `/mnt/c` into `~` |
| Out of disk during image build | Set `APPTAINER_TMPDIR` to a larger location |
| CUDA out of memory | Don't run YOLO + VLM + Ollama together on 6 GB |

---

## 10. Prompt for Claude Code on the tower PC

Clone the repo on the tower, open Claude Code in that folder (inside WSL), and give it the prompt below.

```text
You are helping me get the LASR Base repo (ROS 2 Humble, TIAGo simulation) running on this Windows PC using WSL2 + Apptainer with my NVIDIA GPU.

Read documentation/WSL_APPTAINER_SETUP.md first. It is untested, so treat it as a plan, not as truth. Also read README.md, ros2.def and common/simulation/README.md.

Work through the guide step by step:
1. Inspect the environment: WSL version, Ubuntu release, `nvidia-smi`, free disk and RAM, whether apptainer is installed, whether tiago_humble_os.sif exists and where. Report what is missing before installing or changing anything.
2. Install/configure whatever is missing (Apptainer, etc.). Ask me before anything that needs sudo or large downloads.
3. Fix the `From:` path in ros2.def for my machine (do not commit that change), then build ros2.sif. Run long builds in the background and report errors verbatim.
4. Get `nvidia-smi` and GUI/GL working inside the container (see the WSL notes in the guide). Verify with `glxinfo -B`; software rendering (llvmpipe) is a failure, not a pass.
5. Build the workspace inside the container with colcon and launch `ros2 launch simulation sim.launch.py`. Confirm Gazebo starts and TIAGo spawns. Use topic/service listing as evidence, not assumptions.
6. Only after the sim works, ask which task I want to work on (HRI, GPSR, restaurant, pick_and_place) and bring up just the subsystems it needs, watching VRAM.

Rules:
- Keep everything on the Linux filesystem, not /mnt/c.
- Don't run destructive commands (rm -rf, wsl --unregister, driver changes) without asking.
- When a step fails, show the exact error, diagnose it, and propose a fix before trying it.
- Whenever you find that the guide is wrong or incomplete, update documentation/WSL_APPTAINER_SETUP.md with the verified fix.
- Report what is verified working vs. assumed.
```

---

## 11. Verified notes (✅ tested)

**Disk:** the WSL virtual disk lives on `C:`. When `C:` filled up (< 1 MB free) WSL returned `Input/output error` on `/usr/bin/*` and builds were truncated. Keep 30+ GB free on `C:`, or move the distro: `wsl --manage Ubuntu --move F:\WSL`.

**Memory:** with 16 GB for WSL and heavy Windows apps open, builds were killed for low memory. Setting `memory=20GB` / `swap=16GB` in `.wslconfig` and closing browsers/Docker Desktop fixed it. Use `colcon build --parallel-workers 1` and `MAKEFLAGS=-j2`.

**No `tiago_humble_os.sif`?** PAL does not publish one. A working stand-in: `Bootstrap: docker` / `From: osrf/ros:humble-desktop-full`, then `vcs import` `tiago_public.repos` (from `pal-robotics/tiago_tutorials`, `humble-devel`) into `/opt/tiago_public_ws`, `rosdep install`, `colcon build`, and write `/opt/env.sh` (sources ROS + that workspace) and an empty `/opt/pal/alum/setup.bash`, which `ros2.def` sources. Build it as `tiago_humble_os.sif`, then build `ros2.sif` as in step 4. Private PAL packages are not included.

**GPU + GL in the container** (replaces the guesses in step 5). `--nv` works, but `LD_LIBRARY_PATH` must go through `APPTAINERENV_` and Mesa must use D3D12, otherwise `nvidia-smi` fails ("GPU access blocked") and GL falls back to `llvmpipe`:

```bash
export APPTAINERENV_LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/lib:/lib/x86_64-linux-gnu
export APPTAINERENV_GALLIUM_DRIVER=d3d12
export APPTAINERENV_MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
apptainer run --nv -B /usr/lib/wsl -B /tmp/.X11-unix -B /mnt/wslg -B /run/user/$UID ros2.sif
```

Verified inside: `nvidia-smi -L` lists the GPU and `glxinfo -B` reports `D3D12 (NVIDIA GeForce RTX 3060)`. `--nvccli` does not work (no `nvidia-container-cli`).

**Workspace + sim:** inside the container, `source /opt/env.sh`, bind the workspace, `colcon build --packages-up-to simulation`, then `ros2 launch simulation sim.launch.py`. Expect harmless early errors while controllers start. `moveit` logs `chomp_interface/CHOMPPlanner does not exist` (the CHOMP planner package is missing from the image).

**Not verified:** `robot_ui` npm build (the image step fails because `robot_ui` is not in the image workspace), Ollama (skipped), YOLO/VLM/Whisper, and the task packages.
