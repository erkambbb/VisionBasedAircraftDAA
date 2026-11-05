"""Overlay QA: draws YOLO boxes on random frames and prints size statistics.

Usage: python3 qa_dataset.py NAME [--n 30] [--out DIR] [--labels-only]
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.append("..")
import data_generation.constants as c

BINS = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 1e9)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="/tmp/xp_qa")
    ap.add_argument("--labels-only", action="store_true", help="no state_data.csv needed")
    a = ap.parse_args(argv)
    root = os.path.join(c.PATH, a.name)
    imgs = sorted(glob.glob(os.path.join(root, "images", "*", "*.jpg")))
    if not imgs:
        raise SystemExit(f"görüntü yok: {root}")

    def label_of(p):
        sub = os.path.basename(os.path.dirname(p))
        return os.path.join(root, "labels", sub, os.path.splitext(os.path.basename(p))[0] + ".txt")

    widths, empty = [], 0
    for p in imgs:
        lp = label_of(p)
        lines = open(lp).read().split() if os.path.exists(lp) else []
        if len(lines) < 5:
            empty += 1
            continue
        w_img = cv2.imread(p).shape[1] if not widths else None
        widths.append(float(lines[3]))
    img_w = cv2.imread(imgs[0]).shape[1]
    wpx = np.array(widths) * img_w
    print(f"kare: {len(imgs)}  etiketsiz: {empty}")
    if len(wpx):
        print(f"bb_w px (görüntü genişliği {img_w}): min {wpx.min():.1f} medyan {np.median(wpx):.1f} maks {wpx.max():.1f}")
        for lo, hi in BINS:
            print(f"  {lo:>3}-{hi if hi < 1e8 else '+':<4}: {int(((wpx >= lo) & (wpx < hi)).sum())}")
    csv = os.path.join(root, "state_data.csv")
    if not a.labels_only and os.path.exists(csv):
        df = pd.read_csv(csv, skipinitialspace=True)
        if "hfov_meas" in df.columns:
            print("hfov_meas değerleri:", sorted(df["hfov_meas"].unique()))
        for col in ("loc", "clouds"):
            if col in df.columns:
                print(col, dict(df[col].value_counts()))

    random.seed(0)
    sample = random.sample(imgs, min(a.n, len(imgs)))
    tiles = []
    for p in sample:
        im = cv2.imread(p)
        h, w = im.shape[:2]
        lp = label_of(p)
        t = open(lp).read().split() if os.path.exists(lp) else []
        crop = np.zeros((200, 200, 3), np.uint8)
        if len(t) >= 5:
            cx, cy, bw, bh = (float(v) for v in t[1:5])
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)
            m = max(bw * w, bh * h, 20) * 1.5
            xa, ya = int(max(cx * w - m, 0)), int(max(cy * h - m, 0))
            xb, yb = int(min(cx * w + m, w)), int(min(cy * h + m, h))
            if xb > xa and yb > ya:
                crop = cv2.resize(im[ya:yb, xa:xb], (200, 200), interpolation=cv2.INTER_NEAREST)
            cv2.putText(im, f"w={bw*w:.0f}px", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        small = cv2.resize(im, (320, 180))
        crop = cv2.resize(crop, (180, 180))
        tiles.append(np.hstack([small, crop[:, :, :]])[:180])
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    os.makedirs(a.out, exist_ok=True)
    outp = os.path.join(a.out, "overlay_grid.jpg")
    cv2.imwrite(outp, np.vstack(rows))
    print("ızgara:", outp)


if __name__ == "__main__":
    main()
