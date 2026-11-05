"""Environment check: XPC link, X-Plane version, FOV write/readback, frame capture freshness.

Usage: python3 check_xplane.py [--hfov DEG] [--save-dir DIR] [--monitor N]
Exit: 0 ok, 1 connection/version, 2 stale frame, 3 black frame.
"""
import argparse
import os
import sys
import tempfile
import time

import cv2
import mss
import numpy as np

sys.path.append('..')
from xpc3 import XPlaneConnect  # noqa: E402
from data_generation.projection import fov_from_projection  # noqa: E402


def grab(sct, monitor):
    return np.array(sct.grab(sct.monitors[monitor]))[:, :, :3]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hfov", type=float, default=None)
    ap.add_argument("--save-dir", default=None, help="default: a new private temp folder")
    ap.add_argument("--monitor", type=int, default=1)
    a = ap.parse_args(argv)
    a.save_dir = a.save_dir or tempfile.mkdtemp(prefix="xp_check_")
    os.makedirs(a.save_dir, exist_ok=True)
    print(f"kayıt klasörü: {a.save_dir}")

    try:
        client = XPlaneConnect()
        client.socket.settimeout(2)
        ver = client.getDREF("sim/version/xplane_internal_version")[0]
    except Exception as e:
        print(f"HATA: X-Plane'e baglanilamadi ({e}). XPlaneConnect yuklu mu?")
        return 1
    print("X-Plane internal version:", ver)
    if ver <= 110000:
        print("HATA: surum 11.5+ olmali")
        return 1

    client.sendVIEW(85)
    if a.hfov is not None:
        client.sendDREF("sim/graphics/view/field_of_view_deg", a.hfov)
        time.sleep(1.0)
    h, v = fov_from_projection(client.getDREF("sim/graphics/view/projection_matrix_3d"))
    print(f"Olculen FOV: hfov={h:.3f} vfov={v:.3f}")
    if a.hfov is not None and abs(h - a.hfov) > 0.1:
        print("UYARI: FOV hedefi tutmadi")

    with mss.mss() as sct:
        client.pauseSim(True)
        client.sendDREF("sim/operation/override/override_joystick", 1)
        client.sendPOSI([37.46, -122.11, 3048.0, 0, 0, 0], 0)
        time.sleep(1.0)
        fa = grab(sct, a.monitor)
        client.sendPOSI([37.50, -122.00, 3300.0, 10, 5, 120], 0)
        time.sleep(1.0)
        fb = grab(sct, a.monitor)
        client.pauseSim(False)
    cv2.imwrite(os.path.join(a.save_dir, "check_a.png"), fa)
    cv2.imwrite(os.path.join(a.save_dir, "check_b.png"), fb)
    print("Kare boyutu:", fa.shape[1], "x", fa.shape[0])
    if fa.max() == 0 and fb.max() == 0:
        print("HATA: kareler siyah")
        return 3
    if np.array_equal(fa, fb):
        print("BAYAT KARE: yakalama X-Plane'i gormuyor")
        return 2
    print("TAMAM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
