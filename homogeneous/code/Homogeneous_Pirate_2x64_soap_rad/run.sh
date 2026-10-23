#!/bin/bash
# Start this training in the background on ukifg-c091 (no batch system). It keeps running after
# logout. Log: full_time_ext_<start time>.txt in this folder. Stop it with: kill $(cat run.pid)
# If it was interrupted, run this script again: main.py continues from its newest checkpoint.
cd "$(dirname "$0")" || exit 1
if [ -f run.pid ] && kill -0 "$(cat run.pid)" 2>/dev/null; then
    echo "already running, pid $(cat run.pid)"
    exit 1
fi
log=full_time_ext_$(date +%Y%m%d_%H%M%S).txt
PYTHONUNBUFFERED=1 MPLBACKEND=Agg nohup setsid /home/hannan/anaconda3/envs/pinn/bin/python main.py > "$log" 2>&1 < /dev/null &
echo $! > run.pid
echo "started: pid $(cat run.pid), log $log"
