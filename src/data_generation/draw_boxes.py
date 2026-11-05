"""Draws YOLO label boxes on every image of a dataset into a separate folder.

Usage: python3 draw_boxes.py NAME [--out DIR]
Default output: datasets/NAME_boxes/{train,valid}/
"""
import argparse
import glob
import os
import sys

import cv2

sys.path.append("..")
import data_generation.constants as c  # noqa: E402


def draw(img_path, label_path, out_path):
    im = cv2.imread(img_path)
    h, w = im.shape[:2]
    if os.path.exists(label_path):
        for line in open(label_path).read().strip().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in t[1:5])
            x0, y0 = int(round((cx - bw / 2) * w)), int(round((cy - bh / 2) * h))
            x1, y1 = int(round((cx + bw / 2) * w)), int(round((cy + bh / 2) * h))
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 1)
    cv2.imwrite(out_path, im, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = os.path.join(c.PATH, a.name)
    out = a.out or os.path.join(c.PATH, a.name + "_boxes")
    n = 0
    for split in ("train", "valid"):
        os.makedirs(os.path.join(out, split), exist_ok=True)
        for p in sorted(glob.glob(os.path.join(root, "images", split, "*.jpg"))):
            b = os.path.splitext(os.path.basename(p))[0]
            draw(p, os.path.join(root, "labels", split, b + ".txt"), os.path.join(out, split, b + ".jpg"))
            n += 1
    print(f"{n} görüntü yazıldı: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
