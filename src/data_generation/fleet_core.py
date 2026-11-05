"""Pure helpers for the overnight multi-UAV dataset run (no X-Plane imports)."""
import math
import os
import re
import numpy as np

FLEET_ACF_DIR = "Aircraft/Extra Aircraft/uav_fleet"
CAM_HFOV_DEG = 29.36          # IMX477 + 12 mm, 16:9 crop
OUT_WIDTH = 1280
F_OUT_PX = (OUT_WIDTH / 2) / math.tan(math.radians(CAM_HFOV_DEG) / 2)   # ~2444 px

WEATHER_CHOICES = (0, 1, 2)             # 0 clear, 1 cirrus, 2 scattered
WEATHER_PROBS = (0.70, 0.20, 0.10)
ALT_FT = (170.0, 1000.0)
FT = 0.3048
NEAR_FRACTION = 0.10                    # share of frames close to the camera
NEAR_RANGE_M = (0.1, 5.0)               # distance from camera to the nearest surface
FAR_RANGE_M = (5.0, 300.0)
EDGE_FRACTION = 0.15                    # share of frames with the UAV cut by the picture edge
# make_fleet_acfs.py uses the shadow_twin_boom template: its CG (the position X-Plane reports)
# is 3.3 ft aft and 0.3 ft above the OBJ origin. Body points are taken relative to the CG.
CG_OFFSET_M = (0.0, 0.3 * 0.3048, 3.3 * 0.3048)
BIG_MODEL = "uav_166_tactical_talon"
FRAMES_PER_MODEL, FRAMES_BIG_MODEL = 120, 1000


def obj_vertices(obj_path):
    """All VT positions (N,3) of an OBJ8 file, in body frame (x right, y up, z aft)."""
    pts = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("VT "):
                p = line.split()
                pts.append((float(p[1]), float(p[2]), float(p[3])))
    if not pts:
        raise ValueError(f"no VT lines in {obj_path}")
    return np.asarray(pts)


def hull_points(pts):
    """Convex hull vertices (their projection has the same bounding box as the mesh)."""
    try:
        from scipy.spatial import ConvexHull
        return pts[ConvexHull(pts).vertices]
    except Exception:
        mn, mx = pts.min(0), pts.max(0)
        return np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])


class ModelInfo:
    def __init__(self, name, acf_path, hull, radius, extent):
        self.name, self.acf_path, self.hull = name, acf_path, hull
        self.radius = radius      # max distance from the aircraft origin to any vertex
        self.extent = extent      # longest side of the axis-aligned bounds


def load_model(models_dir, name):
    obj = os.path.join(models_dir, name, "objects", f"{name}.obj")
    pts = obj_vertices(obj) - np.asarray(CG_OFFSET_M)
    return ModelInfo(name=name,
                     acf_path=f"{FLEET_ACF_DIR}/{name}/{name}.acf",
                     hull=hull_points(pts),
                     radius=float(np.linalg.norm(pts, axis=1).max()),
                     extent=float((pts.max(0) - pts.min(0)).max()))


def list_models(models_dir):
    names = sorted(d for d in os.listdir(models_dir)
                   if re.match(r"uav_\d+_", d) and os.path.isdir(os.path.join(models_dir, d)))
    return names


def frames_for(name):
    return FRAMES_BIG_MODEL if name == BIG_MODEL else FRAMES_PER_MODEL


def split_counts(n):
    """(train, valid) with about one sixth validation: 120 -> (100, 20), 1000 -> (833, 167)."""
    valid = int(round(n / 6.0))
    return n - valid, valid


def far_limit_m(extent, min_px, hi=FAR_RANGE_M[1]):
    """Largest surface distance at which the longest side still spans min_px output pixels."""
    return float(min(hi, max(FAR_RANGE_M[0] + 1.0, F_OUT_PX * extent / min_px)))


def sample_distance(rng, radius, extent, min_px=8.0):
    """(d_surface, d_center) in meters. d_center = d_surface + radius (aircraft origin to camera)."""
    if rng.random() < NEAR_FRACTION:
        lo, hi = NEAR_RANGE_M
    else:
        lo, hi = FAR_RANGE_M[0], far_limit_m(extent, min_px)
    d = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return d, d + radius


def edge_angles(rng, hang, vang, hfov, vfov, raw_hfov, d_center, radius):
    """(hang, vang) with the UAV on a random edge of the camera picture: partly in, partly out,
    like an aircraft entering or leaving the frame. Stays inside the wider raw frame."""
    spread = math.degrees(math.atan2(radius, d_center)) * rng.uniform(-0.8, 0.8)
    side = 1.0 if rng.random() < 0.5 else -1.0
    if rng.random() < 0.5:
        return float(np.clip(side * (hfov / 2 + spread), -raw_hfov / 2, raw_hfov / 2)), vang
    raw_vhalf = math.degrees(math.atan(math.tan(math.radians(raw_hfov) / 2) * 9 / 16))
    return hang, float(np.clip(side * (vfov / 2 + spread), -raw_vhalf, raw_vhalf))


def sample_altitude_m(rng):
    return float(rng.uniform(*ALT_FT)) * FT


def pick_weather(rng):
    return int(rng.choice(WEATHER_CHOICES, p=WEATHER_PROBS))


def bbox_out_px(bbox, sw, xp_hfov):
    """(w, h) of a raw-frame bbox in output-frame pixels (1280 wide, camera FOV)."""
    f_raw = (sw / 2) / math.tan(math.radians(xp_hfov) / 2)
    k = F_OUT_PX / f_raw
    return bbox[2] * k, bbox[3] * k
