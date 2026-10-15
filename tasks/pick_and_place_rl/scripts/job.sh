#!/bin/bash
# usage: job.sh <logfile> <python module> [args...]
# Fresh sim -> run one module -> stop the sim. Never runs two controllers at once.
# Kills stale ROS nodes first (a leftover play_motion2 breaks the arm tuck) and retries the sim start.
LOG=$1; shift
cleanup() {
  pkill -f "[r]os2 launch"; pkill -f "u -m [p]ick_and_place_rl"; sleep 2
  pkill -9 -f "[g]zserver"; pkill -9 -f "[g]zclient"; pkill -9 -f "[p]lay_motion2"
  pkill -9 -f "[t]uck_arm"; pkill -9 -f "[r]os-args"; sleep 8
}
for attempt in 1 2 3 4; do
  cleanup
  : > ~/ros_ws/sim_live.log
  ~/ros_ws/start_sim.sh &
  for i in $(seq 1 120); do grep -q "Arm tucked" ~/ros_ws/sim_live.log 2>/dev/null && break; sleep 2; done
  grep -q "Arm tucked" ~/ros_ws/sim_live.log && break
  echo "sim start attempt $attempt failed (arm not tucked)" >&2
done
grep -q "Arm tucked" ~/ros_ws/sim_live.log || { echo "SIM FAILED" > "$LOG"; echo "JOB FINISHED" >> "$LOG"; exit 1; }
~/ros_ws/run_lasr.sh exec bash -c "source /opt/env.sh >/dev/null; export PYTHONPATH=/ros_ws/src/Base/tasks/pick_and_place_rl:\$PYTHONPATH; cd /ros_ws; python3 -u -m $*" 2>&1 | grep --line-buffered -v -E "^I am|INFO:" > "$LOG"
cleanup
echo "JOB FINISHED" >> "$LOG"
