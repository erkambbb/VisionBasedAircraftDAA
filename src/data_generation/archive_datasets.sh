#!/usr/bin/env bash
# Moves old datasets into datasets/arsiv/ and colours the folders (Nemo custom icons).
# Usage: ./archive_datasets.sh [NAME]   NAME = current dataset, never moved, coloured once it exists.
# Safe to run again: nothing is overwritten. DATASETS_DIR selects another folder (used for testing).
cd "$(dirname "$0")" || exit 1
ROOT=$(realpath "${DATASETS_DIR:-../../datasets}") || exit 1
NAME=${1:-uav_fleet_v3}
ARSIV=$ROOT/arsiv
mkdir -p "$ARSIV" || exit 1
shopt -s nullglob

colour() {   # colour <Mint-Y colour> <dir>...
    local icon="file:///usr/share/icons/Mint-Y-$1/places/48/folder.png" d
    shift
    for d in "$@"; do
        [ -d "$d" ] && { gio set -t string "$d" metadata::custom-icon "$icon" 2>/dev/null || echo "renk verilemedi: $d"; }
    done
}

for d in "$ROOT"/uav_fleet_v1* "$ROOT"/uav_fleet_v2* "$ROOT"/xp12mm_* "$ROOT"/sample_mini "$ROOT"/sample_small "$ROOT"/pt; do
    [ -d "$d" ] || continue
    base=$(basename "$d")
    case "$base" in "$NAME"|"$NAME"_*) continue ;; esac
    if [ -e "$ARSIV/$base" ]; then
        echo "atlandı, arşivde zaten var: $base"
        continue
    fi
    mv "$d" "$ARSIV/" || exit 1
done

colour Orange "$ARSIV"
colour Sand "$ARSIV"/uav_fleet_v1*
colour Teal "$ARSIV"/uav_fleet_v2*
colour Navy "$ARSIV"/xp12mm_*
colour Grey "$ARSIV"/sample_mini "$ARSIV"/sample_small "$ARSIV"/pt
colour Blue "$ROOT/$NAME"
colour Aqua "$ROOT/${NAME}_cam"
colour Red "$ROOT/${NAME}_cam_boxes"
