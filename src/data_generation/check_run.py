"""Status of a fleet run, for the 15 minute check. Reads file names only, never images.

    python3 check_run.py [NAME]        # default uav_fleet_v3
Prints a report; the last line is DURUM: tamam or DURUM: SORUN (...). Exit code 1 on SORUN.
"""
import collections
import json
import os
import re
import subprocess
import sys
import time

DATASETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets")
BIG_MODEL, FRAMES_MODEL, FRAMES_BIG, N_MODELS = "uav_166_tactical_talon", 120, 1000, 166
STALE_S = 300                      # alive but no new frame for this long = stuck


def expected(name):
    return FRAMES_BIG if name == BIG_MODEL else FRAMES_MODEL


def frame_counts(root):
    """({model index: saved frames}, age in seconds of the newest frame or None)."""
    counts, newest = collections.Counter(), None
    for split in ("train", "valid"):
        d = os.path.join(root, "images", split)
        if not os.path.isdir(d):
            continue
        with os.scandir(d) as it:
            for e in it:
                m = re.match(r"u(\d{3})_\d+\.jpg$", e.name)
                if m:
                    counts[int(m.group(1))] += 1
                    t = e.stat().st_mtime
                    newest = t if newest is None or t > newest else newest
    return counts, None if newest is None else time.time() - newest


def log_summary(root, n=6):
    """(last n log lines without the 'cevap vermedi' noise, number of KONTROL lines, number of SORUN lines)."""
    path = os.path.join(root, "fleet_run.log")
    if not os.path.exists(path):
        return [], 0, 0
    lines = [l.rstrip() for l in open(path, errors="ignore")]
    checks = [l for l in lines if "KONTROL:" in l]
    return ([l for l in lines if "cevap vermedi" not in l][-n:], len(checks), sum("SORUN" in l for l in checks))


def running(args):
    return subprocess.run(["pgrep"] + args, capture_output=True).returncode == 0


def report(root, gen_alive, xp_alive):
    """(report lines, list of problems)."""
    prog_path = os.path.join(root, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"done": [], "failed": []}
    counts, age = frame_counts(root)
    tail, n_checks, n_bad = log_summary(root)
    finished = len(prog["done"]) + len(prog["failed"]) >= N_MODELS
    total = (N_MODELS - 1) * FRAMES_MODEL + FRAMES_BIG
    wrong = [n for n in prog["done"] if counts[int(n.split("_")[1])] != expected(n)]

    problems = []
    if prog["failed"]:
        problems.append(f"başarısız model: {prog['failed']}")
    if wrong:
        problems.append(f"kare sayısı yanlış: {wrong[:5]}")
    if n_bad:
        problems.append(f"{n_bad} kontrol sorunlu")
    if not finished:
        if not gen_alive:
            problems.append("üretim çalışmıyor")
        if not xp_alive:
            problems.append("X-Plane kapalı")
        if gen_alive and age is not None and age > STALE_S:
            problems.append(f"{age / 60:.0f} dakikadır yeni kare yok")

    montage = os.path.join(root, "kontrol", "son_kontrol.jpg")
    lines = [f"üretim: {'çalışıyor' if gen_alive else 'yok'}, X-Plane: {'açık' if xp_alive else 'kapalı'}",
             f"kare: {sum(counts.values())} / {total}, model: {len(prog['done'])} / {N_MODELS}, başarısız: {len(prog['failed'])}",
             "son kare: " + ("yok" if age is None else f"{age:.0f} sn önce"),
             f"KONTROL satırı: {n_checks}, sorunlu: {n_bad}"]
    if os.path.exists(montage):
        lines.append(f"kontrol görüntüsü: {os.path.abspath(montage)} ({(time.time() - os.path.getmtime(montage)) / 60:.0f} dk önce)")
    else:
        lines.append("kontrol görüntüsü: henüz yok")
    return lines + ["son log:"] + ["  " + l for l in tail], problems


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "uav_fleet_v3"
    root = os.path.join(DATASETS, name)
    if not os.path.isdir(root):
        print(f"DURUM: SORUN ({root} yok)")
        return 1
    lines, problems = report(root, running(["-f", f"generate_fleet --name {name}"]), running(["-x", "X-Plane-x86_64"]))
    print("\n".join(lines))
    print("DURUM: " + (f"SORUN ({'; '.join(problems)})" if problems else "tamam"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
