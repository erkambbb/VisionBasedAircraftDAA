#!/usr/bin/env bash
# 166-UAV dataset run. Usage: ./run_fleet_night.sh [NAME]   (default uav_fleet_v3)
# Archives old datasets, then starts fresh, or resumes automatically if datasets/NAME exists.
# Keeps the screen awake while it runs. Log: datasets/NAME/fleet_run.log
cd "$(dirname "$0")" || exit 1
NAME=${1:-uav_fleet_v3}
./archive_datasets.sh "$NAME" || exit 1
EXTRA=()
[ -d "../../datasets/$NAME" ] && EXTRA+=(--resume)
systemd-inhibit --what=idle:sleep --who=uav-fleet --why="dataset generation" \
     nice -n 10 python3 -m generate_fleet --name "$NAME" "${EXTRA[@]}"
rc=$?
./archive_datasets.sh "$NAME"      # colours the folders this run created
exit $rc
