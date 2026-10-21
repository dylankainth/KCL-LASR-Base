# Getting pick_and_place_rl running on the uni laptop (Windows + WSL2) and connected to TIAGo

Handoff note for whoever (person or agent) sets this up. State as of the commit that added it. Read `CLAUDE.md` and
`REAL_ROBOT.md` too; this file only covers what is specific to this laptop and what is still open.

## Situation

* The code was developed and tested in Gazebo on a home PC (WSL2 + Apptainer + RTX 3060). **Nothing has touched the
  real robot.** The closed-loop mode (`--closed-loop`) was tested in Gazebo only, with finger contact and the table
  emulated; see the table in `REAL_ROBOT.md`.
* This laptop: Windows, WSL2 Ubuntu, **no NVIDIA GPU** (not needed: the runtime is numpy + rclpy; skip `--nv`).
* Work happens in a clone **inside WSL** (`~/ros_ws/src/Base`), not under `/mnt/c` (slow, breaks permissions, and
  Windows git can check shell scripts out with CRLF line endings, which bash rejects).

## Steps

1. WSL2 check (`wsl -l -v`, VERSION 2). Install Apptainer (PPA `ppa:apptainer/ppa`, or the `.deb` from the Apptainer
   GitHub releases if the PPA does not support the Ubuntu release). `apptainer --version`.
2. `mkdir -p ~/ros_ws/src && cd ~/ros_ws/src && git clone -b pick-and-place-rl <fork url> Base`
3. Get `ros2.sif` (2.0 GB) from the home PC (USB/OneDrive) to `~/ros_ws/ros2.sif` (keep it on ext4, not `/mnt/c`).
   Do **not** rebuild it unless you must: that needs PAL's non-public `tiago_humble_os.sif` (1.6 GB) and `ros2.def`
   edited so its `From:` line points at that file (do not commit that edit). Ask before copying the base image around.
4. Launch the container with `tasks/pick_and_place_rl/scripts/run_laptop.sh` (binds only what exists, no `--nv`
   without a GPU). `run_lasr.sh` is the home PC's WSL+NVIDIA launcher; do not use it here. `run_laptop.sh` was only
   verified on the home PC; the no-GPU path is untested.
5. Inside: `source /opt/env.sh && cd /ros_ws && colcon build --packages-select pick_and_place_rl && source install/setup.bash`
   (no `--symlink-install`). Then `cd /ros_ws/src/Base/tasks/pick_and_place_rl && python3 -m unittest discover -s test`
   should print `Ran 28 tests ... OK`. That proves the code runs here; it says nothing about the robot.

## Connecting to the robot (the open problem)

Nothing in the repo records the robot's network settings. **Ask the user** for: how the laptop connects (ethernet or
wifi), the robot's IP/hostname, its `ROS_DOMAIN_ID`, and its RMW / DDS configuration (Fast DDS or Cyclone, any
discovery server or static peers). Export `ROS_DOMAIN_ID` / `RMW_IMPLEMENTATION` on the **WSL host** before launching
the container: Apptainer passes host environment variables in.

WSL2's default NAT networking usually blocks ROS 2 multicast discovery. In order:

1. `ping <robot ip>` from inside WSL.
2. `C:\Users\<user>\.wslconfig` with `[wsl2]` / `networkingMode=mirrored`, then `wsl --shutdown` (Windows 11 22H2+).
   Mirrored mode is meant to support multicast; **not verified with DDS to this robot.** A Hyper-V firewall rule
   allowing inbound traffic to WSL may also be needed.
3. Still nothing: unicast discovery (static peers / discovery server) with the robot's DDS settings.
4. Success looks like `ros2 topic list` inside the container showing `/joint_states`, `/head_front_camera/...`.

## Then commission, in order (REAL_ROBOT.md section 3)

`preflight` (read-only; `URDF FK matches TF` must pass) -> `perception_node` + tape-measure check -> `park --dry-run`
-> `calibrate` -> `run_real --expert --approach-only` -> `grip_test` -> first full `--expert` run, slow.
Leave `--closed-loop` off until the plain expert works.

## Rules for an agent doing this

* **Never run anything that moves the arm or base** (`run_real`, `park` without `--dry-run`, `calibrate`, `grip_test`,
  `base_probe`, anything sending to `/cmd_vel` or the trajectory controllers) without the user saying a person is at
  the e-stop and the area is clear. `preflight`, `perception_node`, `topic_probe` and `ros2 topic echo` are read-only.
* Do not loosen speed/step defaults. Do not claim something works on the robot until it has run on the robot.
* Likely first problems: DDS discovery (above); camera topic names or colours differing from the sim (`preflight`
  prints the topics; `BLUE`/`RED` in `perception.py`); the gripper type (the code assumes the PAL gripper, not Hey5).
