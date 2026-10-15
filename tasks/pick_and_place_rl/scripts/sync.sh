#!/bin/bash
# sync Windows repo -> WSL clone (keeps WSL-only ros2.def edits)
rsync -a --delete --exclude .git --exclude ros2.def --exclude '*.sif' --exclude __pycache__ \
  /mnt/c/Users/dylan/Documents/GitHub/KCL-LASR-Base/ ~/ros_ws/src/Base/
