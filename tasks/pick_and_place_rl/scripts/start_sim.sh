bash ~/ros_ws/sync.sh
~/ros_ws/run_lasr.sh exec bash -c '
source /opt/env.sh >/dev/null; cd /ros_ws
colcon build --packages-select simulation gazebo_grasp_attach 2>&1 | tail -3
source install/setup.bash
exec ros2 launch simulation sim.launch.py world_name:=pick_place gzclient:=False
' > ~/ros_ws/sim_live.log 2>&1
