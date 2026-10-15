#!/bin/bash
# Run the LASR container on WSL2 with NVIDIA GPU (CUDA + D3D12 GL) and WSLg GUI.
export APPTAINERENV_LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/lib:/lib/x86_64-linux-gnu
export APPTAINERENV_GALLIUM_DRIVER=d3d12
export APPTAINERENV_MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
OVR=$HOME/ros_ws/src/Base/common/simulation/overrides
exec apptainer "${1:-run}" --nv -B /usr/lib/wsl -B /tmp/.X11-unix -B /mnt/wslg -B /run/user/$UID \
  -B $HOME/ros_ws:/ros_ws \
  -B $OVR/gripper.gazebo.xacro:/opt/tiago_public_ws/src/pal_gripper/pal_gripper_description/urdf/gripper.gazebo.xacro \
  ~/ros_ws/ros2.sif "${@:2}"
