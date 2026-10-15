#!/bin/bash
# usage: job.sh <logfile> <python module> [args...]
# Fresh sim -> run one module -> stop the sim. Never runs two controllers at once.
LOG=$1; shift
pkill -f "[r]os2 launch"; pkill -f "u -m [p]ick_and_place_rl"; sleep 2
pkill -9 -f "[g]zserver"; pkill -9 -f "[g]zclient"; sleep 1
: > ~/ros_ws/sim_live.log
~/ros_ws/start_sim.sh &
until grep -q "Arm tucked" ~/ros_ws/sim_live.log 2>/dev/null; do sleep 2; done
~/ros_ws/run_lasr.sh exec bash -c "source /opt/env.sh >/dev/null; export PYTHONPATH=/ros_ws/src/Base/tasks/pick_and_place_rl:\$PYTHONPATH; cd /ros_ws; python3 -u -m $*" 2>&1 | grep --line-buffered -v -E "^I am|INFO:" > "$LOG"
pkill -f "[r]os2 launch"; sleep 3; pkill -9 -f "[g]zserver"; pkill -9 -f "[g]zclient"
echo "JOB FINISHED" >> "$LOG"
