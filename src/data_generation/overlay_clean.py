"""Removes X-Plane's floating "<Anchor deployed>" tag from a frame.

The tag is drawn next to the UAV, so it is present in the frame with the UAV and absent in the
comparison frame without it. It is found by template matching (a high-passed copy of the tag,
assets/anchor_label_ref.png) and replaced by the pixels of the comparison frame.
"""
import os

import cv2
import numpy as np

REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "anchor_label_ref.png")
# tag + 1-2 px margin inside the reference crop (rows 7..18, cols 6..94)
_ROWS, _COLS = slice(7, 18), slice(6, 94)
PAD = 60                      # lets the tag be matched when it is partly outside the frame
REF_W = 1280                  # the reference was cut from saved 1280x720 frames; raw frames are 1.5x larger
MIN_SCORE = 0.6               # measured on v2 frames: no tag <= 0.57, real tags >= 0.64
MIN_ENERGY = 20.0             # flat sky gives meaningless scores; real tags have 480+ (mean squared high-pass)


def _highpass(gray):
    g = gray.astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 3)


def _template():
    ref = cv2.imread(REF_PATH)
    if ref is None:
        raise FileNotFoundError(REF_PATH)
    return _highpass(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))[_ROWS, _COLS].copy()


_T = None


def find_label(frame, min_score=MIN_SCORE):
    """(x, y, w, h, score) of the tag in a BGR frame of any size, in that frame's pixels, or None."""
    global _T
    if _T is None:
        _T = _template()
    s = frame.shape[1] / REF_W
    small = frame if s == 1 else cv2.resize(frame, (REF_W, round(frame.shape[0] / s)), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(small, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
    hp = _highpass(cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY))
    res = cv2.matchTemplate(hp, _T, cv2.TM_CCOEFF_NORMED)
    h, w = _T.shape
    energy = cv2.boxFilter(hp * hp, -1, (w, h), anchor=(0, 0))[:res.shape[0], :res.shape[1]]
    res[energy < MIN_ENERGY] = -1.0
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < min_score:
        return None
    return (int(round((loc[0] - PAD) * s)), int(round((loc[1] - PAD) * s)),
            int(round(w * s)), int(round(h * s)), float(score))


def remove_label(a, b, avoid_mask=None, min_score=MIN_SCORE):
    """Returns (cleaned_a, status). status: 'none' (no tag), 'removed', or 'overlap'
    (the tag sits on the UAV, so it cannot be replaced without erasing the UAV: reject the frame)."""
    hit = find_label(a, min_score)
    if hit is None:
        return a, "none"
    x, y, w, h, _ = hit
    m = int(np.ceil(2 * a.shape[1] / REF_W))
    x0, y0, x1, y1 = max(x - m, 0), max(y - m, 0), min(x + w + m, a.shape[1]), min(y + h + m, a.shape[0])
    if x1 <= x0 or y1 <= y0:
        return a, "none"
    if np.mean(np.abs(a[y0:y1, x0:x1].astype(np.int16) - b[y0:y1, x0:x1].astype(np.int16))) < 2.0:
        return a, "none"                      # frames agree there: not a real tag
    if avoid_mask is not None and np.any(avoid_mask[y0:y1, x0:x1]):
        return a, "overlap"
    out = a.copy()
    out[y0:y1, x0:x1] = b[y0:y1, x0:x1]
    return out, "removed"


def label_left(frame, max_score=MIN_SCORE):
    """True if a tag is still visible (used as a final check on saved frames)."""
    hit = find_label(frame, min_score=max_score)
    return hit is not None
