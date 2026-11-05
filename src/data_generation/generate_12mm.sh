#!/usr/bin/env bash
# 12 mm IMX477 / Shadow Twin Boom dataset production.
# Prerequisites: XPlaneConnect loaded, ownship Cessna, AI aircraft = shadow_twin_boom, check_xplane.py passes.
set -euo pipefail
cd "$(dirname "$0")"
C='-ac "Shadow Twin Boom" --focal-mm 12 --xp-hfov 34 --out-size 1280x720 -aw'
# Train + valid: 3 locations x 6 weather x (250 train + 25 val) = 4950 frames
printf 'y\n' | eval python3 -m generate_traffic_data $C -nt 250 -nv 25 -l \"Palo Alto\" --name xp12mm_v1 --newac
eval python3 -m generate_traffic_data $C -nt 250 -nv 25 -l \"Boston\"   --name xp12mm_v1 --append
eval python3 -m generate_traffic_data $C -nt 250 -nv 25 -l \"Osh Kosh\" --name xp12mm_v1 --append
# Separate test set: unseen location, 6 weather x (35 + 5) = 240 frames
printf 'y\n' | eval python3 -m generate_traffic_data $C -nt 35 -nv 5 -l \"Reno Tahoe\" --name xp12mm_test --newac
