#!/bin/bash
# Run the LASR container on a laptop: native Linux or WSL2, with or without an NVIDIA GPU. [untested on that laptop]
# Unlike run_lasr.sh this binds only what exists on this machine, so it does not fail on a missing /usr/lib/wsl.
#
#   scripts/run_laptop.sh                         # interactive shell in the container
#   scripts/run_laptop.sh exec bash -c '...'      # one command (the same form job.sh uses)
#
# Layout it expects:  ~/ros_ws/ros2.sif   and   ~/ros_ws/src/Base (this repo).  Override with ROS_WS=... / SIF=...
# Host environment (ROS_DOMAIN_ID, RMW_IMPLEMENTATION, CYCLONEDDS_URI, ...) is passed into the container by Apptainer,
# so export those on the host before calling this.
WS=${ROS_WS:-$HOME/ros_ws}
SIF=${SIF:-$WS/ros2.sif}
[ -f "$SIF" ] || { echo "run_laptop.sh: image not found: $SIF (copy ros2.sif there, or set SIF=...)" >&2; exit 1; }
command -v apptainer >/dev/null || { echo "run_laptop.sh: apptainer is not installed" >&2; exit 1; }

args=(-B "$WS:/ros_ws")
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  args+=(--nv)
  echo "run_laptop.sh: NVIDIA GPU found, passing it through" >&2
else
  echo "run_laptop.sh: no NVIDIA GPU, running without --nv (fine for pick_and_place_rl)" >&2
fi
for d in /usr/lib/wsl /mnt/wslg /tmp/.X11-unix "/run/user/$UID"; do  # GUI / WSL bits, only where they exist
  [ -d "$d" ] && args+=(-B "$d")
done

exec apptainer "${1:-shell}" "${args[@]}" "$SIF" "${@:2}"
