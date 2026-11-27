"""Multi-UAV dataset run for the 12 mm IMX477 camera.

Single-UAV frames: every model in plane_models/ gets 90 (uav_166_tactical_talon: 1150), 16 000 in all.
Multi-UAV frames: 100 groups x 20 with 2 UAVs, 100 groups x 20 with 3-4 UAVs (4000 frames).
Background frames: 100 groups x 20 without any UAV (2000 frames, empty label files). 22 000 in all.
Every unit gets the same mix of weather, distance, place, time and altitude (fleet_core.unit_plan).
Frames: 17 cities worldwide (fleet_core.CITIES), blocks of about 30 frames per city, summer, 08-18 local, mixed weather (fleet_core.WEATHER_*),
ownship 150-1000 ft above ground, UAV surface 0.25-300 m away (fleet_core.DIST_BANDS).

Run from src/data_generation:
    python3 -m generate_fleet --smoke                 # 3 models x 4 + 2 groups + 1 background group x 4
    python3 -m generate_fleet --name uav_fleet_v4     # full run
    python3 -m generate_fleet --name uav_fleet_v4 --resume

Needs: X-Plane running in a flight anywhere (the run flies to each city; ownship Cessna 172SP, 4 AI aircraft
in the flight setup),
XPlaneConnect and UavSwitch plugins installed, X-Plane window in front on monitor 1.
"""
import sys
sys.path.append('..')

import argparse
import json
import re
import os
import shutil
import subprocess
import time
import types

import cv2
import mss
import numpy as np
from Xlib import X, display

from xpc3 import XPlaneConnect
import data_generation.constants as c
import data_generation.generate_traffic_data as gtd
from data_generation import fleet_core as fc
from data_generation.helpers import Aircraft, make_yaml_file
from data_generation.camera_profile import camera_fov
from data_generation.label_traffic_data import yolo_line
from data_generation import overlay_clean
from data_generation.projection import (fov_from_projection, body_points_ogl,
                                        project_points, bbox_loose)

XP_DIR = os.path.expanduser("~/.local/share/Steam/steamapps/common/X-Plane 11")
CSV_HEADER = ("filename,model,split,loc,clouds,local_time_sec,alt_msl_m,e0,n0,u0,h0,p0,r0,"
              "vang,hang,d_surface,d_center,e1,n1,u1,h1,p1,r1,raw_w,raw_h,"
              "bb_cx,bb_cy,bb_w,bb_h,out_w_px,out_h_px,geo_dx,geo_dy,hfov_meas,n_uav,slot,spot,day_of_year,agl_meas_m,"
              "local_hhmm,ground_m,agl_ft,visibility_m,rain,thunderstorm,cloud_types,cloud_bases_m\n")
LOAD_WAIT_S = 3.0
SCENERY_WAIT_S = 240.0    # longest wait for the scenery after a jump to another city
SCENERY_SETTLE_S = 15.0   # then a little more for buildings and textures
JPEG_MAX = [cv2.IMWRITE_JPEG_QUALITY, 100]
WEATHER_WAIT_S = 2.0      # rain and visibility need a little longer than clouds
BAD_WEATHER_STREAK = 25   # this many unstable frames in a row: the weather state flickers, pick another
NET_RETRY_S = 3.0
MIN_CLEAR_M = 25.0        # intruder must stay this high above the ground (ground safety)


def park(slot):
    """Far away from the camera (sub-pixel): used for the empty frame and for unused AI slots."""
    return Aircraft(slot, 400.0 * slot, 20000.0, 2000.0 + 50.0 * slot, 0, pitch=0, roll=0)


class Log:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        self.f.write(line + "\n")


def reset_client(old):
    """New XPlaneConnect socket: drops late UDP replies that would answer the wrong question."""
    try:
        old.close()
    except Exception:
        pass
    new = XPlaneConnect()
    new.socket.settimeout(10)
    return new


# ---------------------------------------------------------------- window
def find_xplane_window():
    """(x, y, w, h) of the X-Plane window from wmctrl, or None."""
    try:
        out = subprocess.run(["wmctrl", "-lG"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.split(None, 7)
        if len(parts) == 8 and ("X-System" in parts[7] or parts[7].startswith("X-Plane")):
            return tuple(int(v) for v in parts[2:6])
    return None


def focus_xplane():
    """Raises the X-Plane window so nothing covers it (terminal, notifications, ...)."""
    for title in ("X-System", "X-Plane"):
        r = subprocess.run(["wmctrl", "-a", title], capture_output=True)
        if r.returncode == 0:
            return True
    return False


def xplane_window_id():
    """X11 id (hex string) of the X-Plane window, or None."""
    try:
        out = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and ("X-System" in parts[3] or parts[3].startswith("X-Plane")):
            return parts[0]
    return None


_x = {}


def grab_xlib(win_id):
    """BGR frame straight from the X server, several times faster than `import`."""
    if "d" not in _x:
        _x["d"] = display.Display()
    if _x.get("id") != win_id:
        _x["id"], _x["w"] = win_id, _x["d"].create_resource_object("window", int(win_id, 16))
    g = _x["w"].get_geometry()
    r = _x["w"].get_image(0, 0, g.width, g.height, X.ZPixmap, 0xFFFFFFFF)
    return np.ascontiguousarray(np.frombuffer(r.data, np.uint8).reshape(g.height, g.width, 4)[:, :, :3])


def grab_window(win_id):
    """BGR frame of the X-Plane window itself (works while it is on another workspace)."""
    try:
        return grab_xlib(win_id)
    except Exception:
        _x.clear()
    r = subprocess.run(["import", "-window", win_id, "ppm:-"], capture_output=True, timeout=20)
    if r.returncode != 0 or not r.stdout:
        raise OSError("import başarısız: " + r.stderr.decode(errors="ignore")[:120])
    img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError("pencere görüntüsü okunamadı")
    return img


def grab_frame(ctx):
    wid = getattr(ctx, "win_id", None)
    return grab_window(wid) if wid else gtd.capture_frame(ctx.screen, ctx.client)


def xplane_active():
    """True when the X-Plane window is the active (front) window."""
    try:
        act = subprocess.run(["xprop", "-root", "_NET_ACTIVE_WINDOW"], capture_output=True, text=True, timeout=5).stdout
        act_id = int(act.split()[-1], 16)
        for line in subprocess.run(["wmctrl", "-l"], capture_output=True, text=True, timeout=5).stdout.splitlines():
            parts = line.split(None, 3)
            if len(parts) == 4 and ("X-System" in parts[3] or parts[3].startswith("X-Plane")) and int(parts[0], 16) == act_id:
                return True
    except Exception:
        return True      # cannot check: do not block the run
    return False


def wait_front(log, max_wait=1800):
    """Blocks until X-Plane is the front window (raises it again and again)."""
    t0 = time.time()
    warned = False
    while not xplane_active():
        focus_xplane()
        if not warned:
            log("  X-Plane önde değil; öne alınıyor, kare alınmıyor")
            warned = True
        if time.time() - t0 > max_wait:
            raise SystemExit("X-Plane 30 dakikadır öne gelmiyor, duruyorum (--resume ile devam)")
        time.sleep(2)


# ---------------------------------------------------------------- model switching
class Switcher:
    """Talks to the UavSwitch X-Plane plugin through Output/uav_switch/*.txt"""

    def __init__(self, xp_dir):
        self.xp_dir = xp_dir
        self.dir = os.path.join(xp_dir, "Output", "uav_switch")
        os.makedirs(self.dir, exist_ok=True)
        self.n = int(time.time())

    def load(self, acf_path, slot=1, timeout=30.0):
        """Loads acf_path into AI slot `slot`. Returns the path X-Plane reports as loaded;
        raises RuntimeError on failure."""
        if not os.path.isabs(acf_path):
            acf_path = os.path.join(self.xp_dir, acf_path)      # plugin gets the full path
        self.n += 1
        rid = str(self.n)
        ack = os.path.join(self.dir, "ack.txt")
        if os.path.exists(ack):
            os.remove(ack)
        tmp = os.path.join(self.dir, "request.tmp")
        with open(tmp, "w") as f:
            f.write(f"{rid}\t{acf_path}\t{slot}\n")
        os.replace(tmp, os.path.join(self.dir, "request.txt"))
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(ack):
                parts = open(ack).read().strip().split("\t")
                if parts and parts[0] == rid:
                    if len(parts) >= 2 and parts[1] == "ok":
                        loaded = parts[2] if len(parts) > 2 else ""
                        if os.path.basename(acf_path) not in loaded:
                            raise RuntimeError(f"UavSwitch yanlış model bildirdi: {loaded}")
                        return loaded
                    raise RuntimeError("UavSwitch: " + " ".join(parts[1:]))
            time.sleep(0.2)
        raise RuntimeError("UavSwitch cevap vermedi: eklenti yüklü mü? (X-Plane Log.txt'te UavSwitch ara)")

    def probe(self, latlons, timeout=120.0):
        """Terrain height (m MSL, nan = not loaded) at each (lat, lon), from the plugin's probe."""
        self.n += 1
        rid = f"p{self.n}"
        ack = os.path.join(self.dir, "probe_ack.txt")
        if os.path.exists(ack):
            os.remove(ack)
        tmp = os.path.join(self.dir, "probe_request.tmp")
        with open(tmp, "w") as f:
            f.write(rid + "\n" + "".join(f"{la:.7f} {lo:.7f}\n" for la, lo in latlons))
        os.replace(tmp, os.path.join(self.dir, "probe_request.txt"))
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(ack):
                lines = open(ack).read().split()
                if lines and lines[0] == rid:
                    return np.array([float(v) for v in lines[1:]])
            time.sleep(0.2)
        raise RuntimeError("UavSwitch arazi yoklaması cevap vermedi (eklenti eski olabilir: X-Plane'i yeniden başlat)")


def wait_scenery(ctx, city):
    """After a jump to another city: waits until X-Plane has loaded the scenery under every spot
    (terrain probes hit), then a little longer for buildings and textures."""
    pts = [(la, lo) for _, la, lo, *_ in fc.SPOTS[city]]
    t0, hit = time.time(), 0.0
    while time.time() - t0 < SCENERY_WAIT_S:
        time.sleep(5.0)
        hit = float(np.mean(~np.isnan(ctx.switcher.probe(pts))))
        if hit >= 0.999:
            break
    ctx.log(f"  {city}: sahne {time.time() - t0:.0f} sn'de yüklendi (noktaların %{hit * 100:.0f}'i)")
    time.sleep(SCENERY_SETTLE_S)


def enter_city(ctx, city):
    """Moves the run to `city`: region origin, spots, airports, terrain grid (cached per city)."""
    if getattr(ctx, "city", None) == city:
        return
    lat0, lon0 = fc.CITIES[city][:2]
    c.REGION_OPTIONS[city] = [lat0, lon0, 0.0]
    gtd.args.location = city
    ctx.city, ctx.spots, ctx.airports = city, fc.spot_offsets(city), fc.airport_offsets(city)
    ctx.log(f"şehir: {city} ({len(ctx.spots)} nokta, {len(ctx.airports)} havalimanından uzak)")
    s = ctx.spots[0]
    gtd.set_position(ctx.client, Aircraft(0, s[1], s[2], s[3] + 300.0, 0, pitch=0, roll=0))
    for slot in range(1, ctx.args.ai_slots + 1):
        gtd.set_position(ctx.client, park(slot))
    wait_scenery(ctx, city)
    slug = re.sub(r"[^a-z0-9]+", "_", city.lower())
    ctx.terrain = build_terrain(ctx, os.path.join(ctx.root, f"terrain_{slug}.npz"))


def build_terrain(ctx, path):
    """Terrain grid over the city's spots, probed once and cached in the dataset folder (resume)."""
    import pymap3d as pm
    if not ctx.spots:
        return None
    if os.path.exists(path):
        z = np.load(path)
        return fc.Terrain(float(z["e0"]), float(z["n0"]), float(z["step"]), z["msl"])
    es, ns = fc.Terrain.grid(ctx.spots)
    ee, nn = np.meshgrid(es, ns)
    lat0, lon0 = fc.region_origin(city_of(ctx))
    lat, lon, _ = pm.enu2geodetic(ee.ravel(), nn.ravel(), 0.0, lat0, lon0, 0.0)
    msl = ctx.switcher.probe(list(zip(lat, lon))).reshape(ee.shape)
    hit = float(np.mean(~np.isnan(msl)))
    ctx.log(f"arazi: {msl.size} nokta yoklandı, %{hit * 100:.0f} bulundu, "
            f"yükseklik {np.nanmin(msl) if hit else 0:.0f}-{np.nanmax(msl) if hit else 0:.0f} m")
    if hit < 0.5:
        ctx.log("  UYARI: arazinin yarısından azı yüklü; eksik yerlerde bölge tahmini zemin kullanılacak")
    np.savez(path, e0=es[0], n0=ns[0], step=es[1] - es[0], msl=msl)
    return fc.Terrain(es[0], ns[0], es[1] - es[0], msl)


# ---------------------------------------------------------------- frame geometry
VIEW_REFS = ["sim/graphics/view/world_matrix", "sim/graphics/view/projection_matrix_3d",
             "sim/operation/misc/frame_rate_period", "sim/flightmodel/position/y_agl"]


def geom_refs(k):
    return [f"sim/multiplayer/position/plane{s}_{a}" for s in range(1, k + 1)
            for a in ("x", "y", "z", "psi", "the", "phi")] + VIEW_REFS


def live_geoms(client, sw, sh, hulls):
    """([(geometric box (cx, cy, w, h) or None, convex polygon of the hull in px)] for AI slots
    1..len(hulls), seconds per rendered frame, ownship height above ground (m)).
    One request: every XPlaneConnect round trip costs a rendered frame."""
    v = client.getDREFs(geom_refs(len(hulls)))
    wm, pm, period = v[-4], v[-3], v[-2][0]
    agl = v[-1][0] if len(v[-1]) else float("nan")
    out = []
    for s, hull in enumerate(hulls):
        x, y, z, psi, the, phi = (r[0] for r in v[6 * s:6 * s + 6])
        pts = body_points_ogl(hull, (x, y, z), psi, the, phi)
        px, front = project_points(pts, wm, pm, sw, sh)
        box = bbox_loose(px, front, sw, sh, min_visible=0.01)
        poly = cv2.convexHull(px.astype(np.float32)).reshape(-1, 2) if box is not None else None
        out.append((box, poly))
    return out, period, agl


def settle(ctx):
    """Waits for X-Plane to draw the new scene: a few rendered frames, not a fixed time."""
    time.sleep(min(max(ctx.settle * ctx.period, MIN_SETTLE_S), MAX_SETTLE_S))


def adapt(ctx, ok):
    """A stale or unsettled capture slows down the next frames; a run of good frames speeds them up."""
    if not ok:
        ctx.settle, ctx.good = min(ctx.settle + 1, MAX_SETTLE_FRAMES), 0
        return
    ctx.good += 1
    if ctx.good >= 25:
        ctx.settle, ctx.good = max(ctx.settle - 1, MIN_SETTLE_FRAMES), 0


def poly_mask(poly, sw, sh, roi):
    """Filled UAV silhouette polygon (frame size, uint8 0/1) grown by a few pixels."""
    m = np.zeros((sh, sw), np.uint8)
    pts = np.clip(np.round(poly), -4 * sw, 4 * sw).astype(np.int32)
    cv2.fillConvexPoly(m, pts, 1)
    grow = int(max(4, 0.12 * max(roi[2], roi[3])))
    cv2.polylines(m, [pts], True, 1, thickness=2 * grow + 1)
    return m


DIFF_THR = 10
EDGE_TOL_PX = 3.0              # every box edge must match the geometric box within this ...
EDGE_TOL_REL = 0.04            # ... or this share of the box size on that axis (raw pixels)
MIN_VISIBLE_PX = 3.0           # a cut UAV needs at least this much on both sides (output pixels)
MAX_OUTSIDE = 0.02             # share of the frame that may differ outside the UAV (scene not settled)
MIN_SETTLE_FRAMES, MAX_SETTLE_FRAMES = 3, 30
MIN_SETTLE_S, MAX_SETTLE_S = 0.08, 1.0
CHECK_EVERY_S = 900            # self check every 15 minutes


def geometric_clipped(geo, sw, sh):
    cx, cy, w, h = geo
    x0, y0, x1, y1 = max(cx - w / 2, 0), max(cy - h / 2, 0), min(cx + w / 2, sw), min(cy + h / 2, sh)
    return ((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 1e-3), max(y1 - y0, 1e-3))


def edges_agree(box, geo, sw, sh):
    """True if the pixel box sits on the geometric box, edge by edge. Two independent sources
    agreeing means no shadow widened the box and no part of the UAV was missed. Where the raw
    frame cuts the UAV the pixels must reach that edge; the other sides may only be smaller,
    because the visible part of a cut UAV can be shorter than the whole UAV."""
    tx = max(EDGE_TOL_PX, EDGE_TOL_REL * geo[2])
    ty = max(EDGE_TOL_PX, EDGE_TOL_REL * geo[3])
    b = (box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2)
    g = (geo[0] - geo[2] / 2, geo[1] - geo[3] / 2, geo[0] + geo[2] / 2, geo[1] + geo[3] / 2)
    tol = (tx, ty, tx, ty)
    cut = (g[0] <= 0.5, g[1] <= 0.5, g[2] >= sw - 0.5, g[3] >= sh - 0.5)
    if not any(cut):
        return all(abs(p - q) <= t for p, q, t in zip(b, g, tol))
    inside = b[0] >= g[0] - tx and b[1] >= g[1] - ty and b[2] <= g[2] + tx and b[3] <= g[3] + ty
    return inside and all(abs(p - q) <= t for p, q, t, c in zip(b, g, tol, cut) if c)


def crop_mask(args, sw, sh):
    """0/1 mask of the raw pixels that reach the camera picture (rpi-cam-style crop + lens distortion)."""
    sys.path.insert(0, os.path.expanduser(args.camstyle_dir))
    from camstyle.core import CamStyleParams, Geometry
    ow, oh = args.out_size
    g = Geometry(sh, sw, CamStyleParams(focal_mm=args.focal_mm, source_hfov_deg=args.xp_hfov,
                                        out_width=ow, out_height=oh))
    ox, oy = g.src_to_out(*np.meshgrid(np.arange(sw, dtype=float), np.arange(sh, dtype=float)))
    return ((ox >= 0) & (ox <= ow - 1) & (oy >= 0) & (oy <= oh - 1)).astype(np.uint8)


def visible_box(a, b, box, roi, mask, crop, sw, sh):
    """Box of the UAV pixels that reach the camera picture, or None if none do. Cutting the whole
    box instead would stay too tall or wide: the part outside the picture still shapes it."""
    if crop is None:
        return box
    x0, y0 = max(int(box[0] - box[2] / 2) - 1, 0), max(int(box[1] - box[3] / 2) - 1, 0)
    x1, y1 = min(int(box[0] + box[2] / 2) + 2, sw), min(int(box[1] + box[3] / 2) + 2, sh)
    if crop[y0:y1, x0:x1].all():
        return box
    return diff_bbox(a, b, roi, sw, sh, mask & crop)


def outside_share(a, b, mask):
    """Share of the frame that differs outside the UAV. High = the two frames show different scenes."""
    d = cv2.absdiff(a, b)
    d = cv2.max(cv2.max(d[:, :, 0], d[:, :, 1]), d[:, :, 2])
    return float(np.count_nonzero((d > DIFF_THR) & (mask == 0))) / d.size


def diff_bbox(a, b, roi, sw, sh, mask=None):
    """Tight box (cx, cy, w, h) of what differs between frame a (UAV) and b (no UAV).

    With `mask` (the grown UAV silhouette) only differences inside it count, so the UAV's ground
    shadow and anything else that appears next to it never widens the box. Without a mask the
    geometric box roi, widened by its own size, is used. Returns None if nothing differs.
    """
    d = cv2.absdiff(a, b).max(axis=2)
    cx, cy, w, h = roi
    if mask is None:
        mx, my = w + 40, h + 40
        x0, x1 = int(max(cx - w / 2 - mx, 0)), int(min(cx + w / 2 + mx, sw))
        y0, y1 = int(max(cy - h / 2 - my, 0)), int(min(cy + h / 2 + my, sh))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        mask = np.zeros_like(d)
        mask[y0:y1, x0:x1] = 1
    m = ((d > DIFF_THR) & (mask > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, ar in enumerate(areas) if ar >= max(3, 0.02 * areas.max())]
    if not keep:
        return None
    xs0 = min(stats[i, cv2.CC_STAT_LEFT] for i in keep)
    ys0 = min(stats[i, cv2.CC_STAT_TOP] for i in keep)
    xs1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in keep)
    ys1 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in keep)
    return ((xs0 + xs1) / 2.0, (ys0 + ys1) / 2.0, float(xs1 - xs0), float(ys1 - ys0))


PLACE_TRIES = 30


def ground_at(terrain, e, n, fallback):
    return terrain.ground(e, n, fallback) if terrain is not None else fallback - (e * e + n * n) / (2 * fc.EARTH_R)


def sample_state(rng, infos, args, spots=None, plan=None, terrain=None, airports=()):
    """Ownship + one intruder per model in `infos` (AI slots 1..k), or None if unusable.
    `plan` (one entry of fleet_core.unit_plan) fixes altitude, city category, camera look and the
    distance band of the first UAV; the others get a random band, so one can be 20 m away and
    another 150 m. Several UAVs never overlap in the picture: their boxes must stay separate.
    Heights are above the real terrain (probe grid) when `terrain` is given."""
    plan = plan or {}
    alt = plan.get("alt") or fc.sample_altitude_m(rng)
    spot, e, n, g_est = fc.sample_spot(rng, spots, args.enrange, plan.get("category"), airports)
    ground = ground_at(terrain, e, n, g_est)
    own = Aircraft(0, e, n, ground + alt, rng.uniform(0, 360), fc.sample_pitch(rng, plan.get("down"), plan.get("own_p")),
                   fc.tilt_value(rng, plan.get("own_r", fc.sample_tilt_cat(rng)), fc.OWN_ROLL_MAX))
    uavs, placed = [], []
    for slot, info in enumerate(infos, 1):
        for _ in range(PLACE_TRIES):
            band = plan.get("band") if slot == 1 else None
            d_surf, d_cen = fc.sample_distance(rng, info.radius, info.extent, args.min_px, band)
            vang = rng.uniform(-args.vfov / 2, args.vfov / 2)
            hang = rng.uniform(-args.hfov / 2, args.hfov / 2)
            if rng.random() < fc.EDGE_FRACTION:
                hang, vang = fc.edge_angles(rng, hang, vang, args.hfov, args.vfov, args.xp_hfov, d_cen, info.radius)
            ang_r = fc.angular_radius_deg(info.radius, d_cen)
            if fc.apart(placed, hang, vang, ang_r):
                break
        else:
            return None
        placed.append((hang, vang, ang_r))
        intr = gtd.get_intruder_position(own, d_cen, hang, vang, rng.uniform(0, 360))
        intr.id = slot
        intr.p, intr.r = fc.uav_attitude(rng, plan.get("uav_p") if slot == 1 else None,
                                         plan.get("uav_r") if slot == 1 else None)
        if intr.u - ground_at(terrain, intr.e, intr.n, g_est) < MIN_CLEAR_M:
            return None
        uavs.append(dict(intr=intr, vang=vang, hang=hang, d_surf=d_surf, d_cen=d_cen))
    return dict(own=own, uavs=uavs, alt=alt, spot=spot, ground=ground)


# ---------------------------------------------------------------- dataset files
def prepare_dataset(args):
    root = os.path.join(c.PATH, args.name)
    if os.path.exists(root) and not args.resume:
        raise SystemExit(f"{root} zaten var. Devam için --resume, yeni çalışma için başka --name ver.")
    for sub in ("images/train", "images/valid", "labels/train", "labels/valid"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    csv_path = os.path.join(root, "state_data.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(CSV_HEADER)
    make_yaml_file(root, args.name)
    prog_path = os.path.join(root, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"done": [], "failed": []}
    return root, csv_path, prog_path, prog


def remove_model_files(root, prefix):
    for sub in ("images/train", "images/valid", "labels/train", "labels/valid"):
        d = os.path.join(root, sub)
        for fn in os.listdir(d):
            if fn.startswith(prefix + "_"):
                os.remove(os.path.join(d, fn))


# ---------------------------------------------------------------- periodic self check
def selfcheck(ctx):
    """Every 15 min: re-read the last saved frames, make sure nothing foreign is on screen
    (X-Plane tag), labels sit inside the frame, and drop a montage into <dataset>/kontrol/."""
    ctx.last_check = time.time()
    tiles, problems, tags = [], [], []
    for split, fn in ctx.recent:
        img = cv2.imread(os.path.join(ctx.root, "images", split, fn + ".jpg"))
        if img is None:
            problems.append(f"{fn}: görüntü okunamadı")
            continue
        h, w = img.shape[:2]
        if overlay_clean.label_left(img):
            tags.append(fn)
        if float(img.std()) < 4.0:
            problems.append(f"{fn}: görüntü düz renk (ekran yanlış olabilir)")
        for line in open(os.path.join(ctx.root, "labels", split, fn + ".txt")).read().splitlines():
            cx, cy, bw, bh = (float(v) for v in line.split()[1:5])
            x0, y0, x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
            if x0 < -1 or y0 < -1 or x1 > w + 1 or y1 > h + 1 or bw <= 0 or bh <= 0:
                problems.append(f"{fn}: kutu çerçeve dışında")
            cv2.rectangle(img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), (0, 0, 255), 1)
        tiles.append(cv2.resize(img, (640, 360)))
    if tiles:
        tiles = tiles[-6:]
        while len(tiles) < 6:
            tiles.append(np.zeros_like(tiles[0]))
        d = os.path.join(ctx.root, "kontrol")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "son_kontrol.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 2]) for i in (0, 2, 4)]))
    if len(tags) >= 3:                                # one hit can be a look-alike; many mean removal is broken
        problems.append(f"{len(tags)} karede 'Anchor deployed' yazısı kalmış: {', '.join(tags[:3])}")
    elif tags:
        ctx.log(f"  KONTROL uyarı: yazı şüphesi {', '.join(tags)} (gözle bak)")
    msg = f"KONTROL: {ctx.total_saved} kare, {ctx.labels_removed} yazı silindi, son {len(ctx.recent)} kare incelendi"
    if problems:
        ctx.log(msg + ", SORUN: " + "; ".join(problems))
        raise SystemExit("Kontrol başarısız: " + "; ".join(problems[:3]))
    ctx.log(msg + ", sorun yok")


# ---------------------------------------------------------------- one unit
def wcols(local_time, st, wp):
    """Readable per-frame columns: clock, ground, height above ground, weather details."""
    hh, mm = divmod(int(local_time) // 60, 60)
    types_ = "/".join(str(int(wp[f"cloud_type[{i}]"])) for i in range(3))
    bases = "/".join(f"{wp[f'cloud_base_msl_m[{i}]']:.0f}" if wp[f"cloud_type[{i}]"] else "-" for i in range(3))
    return (f"{hh:02d}:{mm:02d},{st['ground']:.1f},{st['alt'] / fc.FT:.0f},{wp['visibility_reported_m']:.0f},"
            f"{wp['rain_percent']:.2f},{wp['thunderstorm_percent']:.2f},{types_},{bases}")


def apply_conditions(ctx, pl, weather, wparams, bad_streak):
    """Weather and clock of plan entry `pl`. Weather is only sent when the preset changes (plans
    keep each preset in one block) or when it kept the scene unstable: then new values, same preset."""
    client = ctx.client
    if pl["weather"] != weather or bad_streak >= BAD_WEATHER_STREAK:
        if bad_streak >= BAD_WEATHER_STREAK:
            ctx.log(f"  {weather}: {bad_streak} kare üst üste kararsız, aynı hava yeni değerlerle")
        weather, wparams = pl["weather"], fc.weather_params(ctx.rng, pl["weather"])
        set_weather(client, wparams)
        bad_streak = 0
        time.sleep(WEATHER_WAIT_S)
    client.sendDREF("sim/time/zulu_time_sec", city_zulu(ctx, pl["local_time"]))
    client.sendDREF("sim/time/local_date_days", city_day(ctx, pl["day"]))
    return weather, wparams, bad_streak


def city_of(ctx):
    return getattr(ctx, "city", None) or ctx.args.location


def city_zulu(ctx, local_sec):
    city = city_of(ctx)
    if city in fc.CITIES:
        return fc.zulu_sec(local_sec, city)
    return (local_sec + (c.TIME_OPTIONS[city] - 1) * 3600) % 86400     # old regions: US summer time


def city_day(ctx, day):
    city = city_of(ctx)
    return fc.city_day(city, day) if city in fc.CITIES else int(day)


def set_weather(client, params):
    for k, v in params.items():
        client.sendDREF("sim/weather/" + k, v)


def load_unit(ctx, infos):
    """Loads the unit's models into AI slots 1..k (skips slots that already hold the model) and
    parks the unused slots far away, so no unlabelled aircraft can show up in a frame."""
    loaded = []
    for slot, info in enumerate(infos, 1):
        if ctx.loaded.get(slot) != info.acf_path:
            ctx.loaded.pop(slot, None)
            loaded.append(ctx.switcher.load(info.acf_path, slot))
            ctx.loaded[slot] = info.acf_path
    for slot in range(len(infos) + 1, ctx.args.ai_slots + 1):
        gtd.set_position(ctx.client, park(slot))
    return loaded


def run_unit(ctx, key, prefix, infos, n, split="train"):
    """Frames for one unit: a single model (infos = [info]) or a multi-UAV group (2-4 models).
    Every UAV of a saved frame has its own checked box; one bad box rejects the whole frame.
    All frames of the unit go to `split` (fleet_core.valid_units picks the valid units)."""
    args, client, log, rng = ctx.args, ctx.client, ctx.log, ctx.rng
    k = len(infos)
    remove_model_files(ctx.root, prefix)

    loaded = load_unit(ctx, infos)
    log(f"[{key}] {', '.join(i.name for i in infos)}: yüklendi ({len(loaded)} yeni), {n} kare "
        f"({split}), yarıçap {', '.join(f'{i.radius:.2f}' for i in infos)} m")
    if loaded:
        time.sleep(LOAD_WAIT_S)

    hulls = [i.hull for i in infos]
    rows, offs, saved, tries = [], [], 0, 0
    rejects = {"ground": 0, "bbox": 0, "small": 0, "net": 0, "label": 0, "mismatch": 0, "unstable": 0, "overlap": 0}
    weather, wparams, bad_streak = None, None, 0
    net_fail = 0
    max_tries = n * (10 if k == 1 else 30)
    plan = fc.unit_plan(rng, n, args.daystart, args.dayend)
    while saved < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"{key}: {tries} denemede {saved}/{n} kare, çok fazla ret {rejects}")
        pl = plan[saved]
        st = sample_state(rng, infos, args, ctx.spots, pl, ctx.terrain, ctx.airports)
        if st is None:
            rejects["ground"] += 1
            continue
        try:
            weather, wparams, bad_streak = apply_conditions(ctx, pl, weather, wparams, bad_streak)
            local_time, day = pl["local_time"], city_day(ctx, pl["day"])
            gtd.set_position(client, st["own"])
            for u in st["uavs"]:
                gtd.set_position(client, u["intr"])
            settle(ctx)
            geoms, ctx.period, agl = live_geoms(client, ctx.sw, ctx.sh, hulls)
            if any(g[0] is None for g in geoms):
                rejects["bbox"] += 1
                continue
            if any(max(fc.bbox_out_px(g[0], ctx.sw, ctx.hfov_meas)) < args.min_px for g in geoms):
                rejects["small"] += 1
                continue
            masks = [poly_mask(poly, ctx.sw, ctx.sh, geo) for geo, poly in geoms]
            union = masks[0].copy()
            overlap = False
            for m in masks[1:]:
                overlap = overlap or bool(np.any(union & m))
                union |= m
            if overlap:
                rejects["overlap"] += 1               # two UAVs touch in the picture: no clean boxes
                continue
            if not getattr(ctx, 'win_id', None):
                wait_front(log)
            frame = grab_frame(ctx)
            # same camera, UAVs moved far away -> pixel difference is the UAVs
            for slot in range(1, k + 1):
                gtd.set_position(client, park(slot))
            settle(ctx)
            empty = grab_frame(ctx)
            if not getattr(ctx, 'win_id', None) and not xplane_active():
                rejects['notfront'] = rejects.get('notfront', 0) + 1
                continue
            net_fail = 0
        except (OSError, TimeoutError, ValueError) as e:
            rejects["net"] += 1
            net_fail += 1
            client = ctx.client = reset_client(ctx.client)   # a late reply would shift all later answers
            log(f"  X-Plane cevap vermedi ({e}); yeniden deneniyor ({net_fail}/200)")
            if net_fail >= 200:
                raise SystemExit("X-Plane 10 dakikadır cevap vermiyor, duruyorum (ilerleme kaydedildi, --resume ile devam edilir)")
            time.sleep(NET_RETRY_S)
            continue

        label = "none"
        for _ in range(k):                            # one X-Plane tag per UAV at most
            frame, status = overlay_clean.remove_label(frame, empty, union)
            if status == "overlap":
                label = "overlap"
                break
            if status == "none":
                break
            ctx.labels_removed += 1
        if label == "overlap" or overlay_clean.label_left(frame):
            rejects["label"] += 1                     # the tag sits on a UAV: cannot be erased cleanly
            continue
        if outside_share(frame, empty, union) > MAX_OUTSIDE:
            rejects["unstable"] += 1                  # scene still changing between the two frames
            bad_streak += 1
            adapt(ctx, False)
            continue
        bad_streak = 0

        boxes, why = [], None
        for (geo, _), mask in zip(geoms, masks):
            bbox = diff_bbox(frame, empty, geo, ctx.sw, ctx.sh, mask)
            if bbox is None:
                why = "nodiff"
                break
            if not edges_agree(bbox, geometric_clipped(geo, ctx.sw, ctx.sh), ctx.sw, ctx.sh):
                why = "mismatch"                      # pixels and geometry disagree (or a building hides it)
                break
            if max(fc.bbox_out_px(bbox, ctx.sw, ctx.hfov_meas)) < args.min_px:
                why = "small"
                break
            off = (bbox[0] - geo[0], bbox[1] - geo[1])
            vis = visible_box(frame, empty, bbox, geo, mask, ctx.crop, ctx.sw, ctx.sh)
            if vis is None or min(fc.bbox_out_px(vis, ctx.sw, ctx.hfov_meas)) < MIN_VISIBLE_PX:
                why = "outside"                       # a UAV that is not in the camera picture
                break
            boxes.append((vis, off))
        if why:
            rejects[why] = rejects.get(why, 0) + 1
            if why == "nodiff":
                adapt(ctx, False)
                if saved == 0 and rejects["nodiff"] >= 30:
                    raise RuntimeError(f"{key}: UAV ekranda görünmüyor (model yüklenmemiş olabilir)")
            if why == "mismatch" and saved == 0 and rejects["mismatch"] >= 60 * k:
                raise RuntimeError(f"{key}: piksel kutusu geometriyle hiç uyuşmuyor")
            continue

        fn = f"{prefix}_{saved:03d}"
        cv2.imwrite(os.path.join(ctx.root, "images", split, fn + ".jpg"), gtd.resize_frame(frame, args.out_size), JPEG_MAX)
        with open(os.path.join(ctx.root, "labels", split, fn + ".txt"), "w") as f:
            f.writelines(yolo_line(*b, ctx.sw, ctx.sh) for b, _ in boxes)
        o = st["own"]
        for slot, (info, u, (bbox, off)) in enumerate(zip(infos, st["uavs"], boxes), 1):
            i = u["intr"]
            ow, oh = fc.bbox_out_px(bbox, ctx.sw, ctx.hfov_meas)
            offs.append(off)
            rows.append(f"{fn}.jpg,{info.name},{split},{city_of(ctx)},{weather},{local_time:.1f},{o.u:.1f},"
                        f"{o.e:.2f},{o.n:.2f},{o.u:.2f},{o.h:.2f},{o.p:.2f},{o.r:.2f},{u['vang']:.3f},{u['hang']:.3f},"
                        f"{u['d_surf']:.3f},{u['d_cen']:.3f},{i.e:.2f},{i.n:.2f},{i.u:.2f},{i.h:.2f},{i.p:.2f},{i.r:.2f},"
                        f"{ctx.sw},{ctx.sh},{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f},{ow:.1f},{oh:.1f},"
                        f"{off[0]:.1f},{off[1]:.1f},{ctx.hfov_meas},{k},{slot},{st['spot']},{day},{agl:.1f},"
                        f"{wcols(local_time, st, wparams)}\n")
        ctx.recent = (ctx.recent + [(split, fn)])[-12:]
        saved += 1
        ctx.total_saved += 1
        adapt(ctx, True)

        if time.time() - ctx.last_check >= CHECK_EVERY_S:
            selfcheck(ctx)

    with open(ctx.csv_path, "a") as f:
        f.writelines(rows)
    mo = np.mean(np.abs(np.array(offs)), axis=0) if offs else (0, 0)
    log(f"[{key}] {saved} kare tamam, denemeler {tries}, retler {rejects}, "
        f"geometri-piksel kayma ort. dx={mo[0]:.1f} dy={mo[1]:.1f} px")


def run_background(ctx, key, n, split="train", phantom=None):
    """Frames without any UAV (empty label file): the detector learns the background. Same plan,
    weather, places and camera as the UAV frames. Each frame is set up as a UAV frame: `phantom()`
    gives a model, and its UAV is placed with the same distance band, edge and ground checks, but it
    is never shown (all AI slots stay parked). So the camera states (altitude, look-down, tilt) follow
    the UAV frames exactly, and no camera state means "no UAV". Two captures must agree, so nothing
    moving (birds, a half-loaded scene) is saved, and no X-Plane tag may be on screen."""
    args, client, log, rng = ctx.args, ctx.client, ctx.log, ctx.rng
    remove_model_files(ctx.root, key)
    for slot in range(1, args.ai_slots + 1):
        gtd.set_position(client, park(slot))
    log(f"[{key}] arka plan: {n} kare ({split})")
    plan = fc.unit_plan(rng, n, args.daystart, args.dayend)
    rows, saved, tries, net_fail = [], 0, 0, 0
    rejects = {"ground": 0, "net": 0, "label": 0, "unstable": 0}
    weather, wparams, bad_streak = None, None, 0
    no_uav = np.zeros((ctx.sh, ctx.sw), np.uint8)
    while saved < n:
        tries += 1
        if tries > n * 10:
            raise RuntimeError(f"{key}: {tries} denemede {saved}/{n} kare, çok fazla ret {rejects}")
        pl = plan[saved]
        ghost = [phantom()] if phantom else []        # placed like a real UAV, never shown
        st = sample_state(rng, ghost, args, ctx.spots, pl, ctx.terrain, ctx.airports)
        if st is None:
            rejects["ground"] += 1
            continue
        try:
            weather, wparams, bad_streak = apply_conditions(ctx, pl, weather, wparams, bad_streak)
            gtd.set_position(client, st["own"])
            settle(ctx)
            _, ctx.period, agl = live_geoms(client, ctx.sw, ctx.sh, [])
            a = grab_frame(ctx)
            settle(ctx)
            b = grab_frame(ctx)
            net_fail = 0
        except (OSError, TimeoutError, ValueError) as e:
            rejects["net"] += 1
            net_fail += 1
            client = ctx.client = reset_client(ctx.client)
            log(f"  X-Plane cevap vermedi ({e}); yeniden deneniyor ({net_fail}/200)")
            if net_fail >= 200:
                raise SystemExit("X-Plane 10 dakikadır cevap vermiyor, duruyorum (ilerleme kaydedildi, --resume ile devam edilir)")
            time.sleep(NET_RETRY_S)
            continue
        if overlay_clean.label_left(a):
            rejects["label"] += 1
            continue
        if outside_share(a, b, no_uav) > MAX_OUTSIDE:
            rejects["unstable"] += 1
            bad_streak += 1
            adapt(ctx, False)
            continue
        bad_streak = 0
        fn = f"{key}_{saved:03d}"
        cv2.imwrite(os.path.join(ctx.root, "images", split, fn + ".jpg"), gtd.resize_frame(a, args.out_size), JPEG_MAX)
        open(os.path.join(ctx.root, "labels", split, fn + ".txt"), "w").close()
        o = st["own"]
        vals = dict(filename=f"{fn}.jpg", model="none", split=split, loc=city_of(ctx), clouds=weather,
                    local_time_sec=f"{pl['local_time']:.1f}", alt_msl_m=f"{o.u:.1f}", e0=f"{o.e:.2f}", n0=f"{o.n:.2f}",
                    u0=f"{o.u:.2f}", h0=f"{o.h:.2f}", p0=f"{o.p:.2f}", r0=f"{o.r:.2f}", raw_w=ctx.sw, raw_h=ctx.sh,
                    hfov_meas=ctx.hfov_meas, n_uav=0, slot=0, spot=st["spot"], day_of_year=city_day(ctx, pl["day"]),
                    agl_meas_m=f"{agl:.1f}")
        head = CSV_HEADER.strip().split(",")
        tail = wcols(pl["local_time"], st, wparams).split(",")
        rows.append(",".join(str(vals.get(h, "")) for h in head[:-len(tail)]) + "," + ",".join(tail) + "\n")
        ctx.recent = (ctx.recent + [(split, fn)])[-12:]
        saved += 1
        ctx.total_saved += 1
        adapt(ctx, True)
        if time.time() - ctx.last_check >= CHECK_EVERY_S:
            selfcheck(ctx)
    with open(ctx.csv_path, "a") as f:
        f.writelines(rows)
    log(f"[{key}] {saved} kare tamam, denemeler {tries}, retler {rejects}")


def run_model(ctx, idx, name, info):
    """Single-UAV frames of one model."""
    n = ctx.args.frames if ctx.args.frames else fc.frames_for(name)
    run_unit(ctx, f"{idx:03d}", f"u{idx:03d}", [info], n)


# ---------------------------------------------------------------- post processing
def make_shards(src, tmp, n):
    """Spreads a dataset over n folders of symlinks so camstyle can run on them in parallel."""
    shards = [os.path.join(tmp, str(i)) for i in range(n)]
    for split in ("train", "valid"):
        for sh in shards:
            os.makedirs(os.path.join(sh, "images", split))
            os.makedirs(os.path.join(sh, "labels", split))
        for k, fn in enumerate(sorted(os.listdir(os.path.join(src, "images", split)))):
            sh = shards[k % n]
            lab = os.path.splitext(fn)[0] + ".txt"
            os.symlink(os.path.join(src, "images", split, fn), os.path.join(sh, "images", split, fn))
            os.symlink(os.path.join(src, "labels", split, lab), os.path.join(sh, "labels", split, lab))
    return shards


def drop_unlabelled(root):
    """Deletes frames whose box fell outside the camera crop. Returns how many were deleted."""
    n = 0
    for split in ("train", "valid"):
        for fn in os.listdir(os.path.join(root, "images", split)):
            if fn.startswith("bg_"):
                continue                              # background frames have no UAV on purpose
            lab = os.path.join(root, "labels", split, os.path.splitext(fn)[0] + ".txt")
            if not os.path.exists(lab) or not open(lab).read().strip():
                os.remove(os.path.join(root, "images", split, fn))
                if os.path.exists(lab):
                    os.remove(lab)
                n += 1
    return n


def post_process(args, log):
    """Camera look (rpi-cam-style, several processes) + red 1 px box overlays."""
    cam_name = args.name + "_cam"
    log(f"kamera dönüşümü: {args.name} -> {cam_name}")
    ref = os.path.expanduser(args.reference)
    src = os.path.join(os.path.abspath(c.PATH), args.name)
    cam = os.path.join(os.path.abspath(c.PATH), cam_name)
    tmp = os.path.join(src, "_shards")
    shutil.rmtree(tmp, ignore_errors=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    procs = []
    for i, shard in enumerate(make_shards(src, tmp, args.cam_workers)):
        cmd = ["nice", "-n", "10", sys.executable, "-m", "camstyle", shard, cam,
               "--focal-mm", "12", "--source-hfov", str(args.xp_hfov), "--out-size", "1280x720",
               "--reference", ref, "--ignore-top", "45", "--match-strength", "0.5",
               "--wb", "1.05,1.0,0.93", "--haze", "0", "--haze-uniform", "0.07",
               "--saturation", "1.0", "--contrast", "0.95", "--blur", "0.4",
               "--photon-scale", "1500", "--read-noise", "0.006", "--jpeg-quality", "70",
               "--jitter", "0.5", "--motion-blur", "3", "--motion-prob", "0.5",
               "--target-blur", "5", "--target-prob", "0.5", "--seed", str(i), "--min-visible", "0"]
        out = open(shard + ".log", "w")
        procs.append((shard, out, subprocess.Popen(cmd, cwd=os.path.expanduser(args.camstyle_dir), env=env,
                                                   stdout=out, stderr=subprocess.STDOUT)))
    bad = False
    for shard, out, p in procs:
        rc = p.wait()
        out.close()
        tail = open(shard + ".log", errors="ignore").read().strip().splitlines()[-1:]
        log(f"camstyle {os.path.basename(shard)}: {tail[0] if tail else 'çıktı yok'}")
        bad = bad or rc != 0
    if bad:
        log(f"camstyle HATA: ayrıntı {tmp}/*.log")
        return
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"kamera kadrajı dışında kalan {drop_unlabelled(cam)} kare silindi")
    make_yaml_file(os.path.join(c.PATH, cam_name), cam_name)
    subprocess.run([sys.executable, "draw_boxes.py", cam_name], check=False)
    log("bitti: " + cam)


# ---------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="uav_fleet_v4")
    ap.add_argument("--models-dir", default=os.path.expanduser("~/Belgeler/ARIKOVANI/plane_models"))
    ap.add_argument("--xp-dir", default=XP_DIR)
    ap.add_argument("--cities", default="all", help="all = every city in fleet_core.CITIES, or a comma "
                    "list (e.g. 'London,Tokyo'); none = flat area around --location")
    ap.add_argument("--location", default="New York", help="region for --cities none (flat area +-enrange)")
    ap.add_argument("--enrange", type=float, default=2500.0, help="ownship east/north range (m) without city spots")
    ap.add_argument("--ai-slots", type=int, default=fc.MAX_UAV, help="AI aircraft in the X-Plane flight setup")
    ap.add_argument("--group-frames", type=int, default=fc.GROUP_FRAMES, help="frames per multi-UAV group")
    ap.add_argument("--no-multi", action="store_true", help="single-UAV frames only")
    ap.add_argument("--only-multi", action="store_true", help="multi-UAV groups only")
    ap.add_argument("--bg-units", type=int, default=fc.BG_UNITS, help="background-only groups (no UAV), group-frames each")
    ap.add_argument("--focal-mm", type=float, default=12.0)
    ap.add_argument("--xp-hfov", type=float, default=c.XP_RENDER_HFOV)
    ap.add_argument("--out-size", default="1280x720")
    ap.add_argument("--monitor", type=int, default=1)
    ap.add_argument("--region", default=None)
    ap.add_argument("--workspace", type=int, default=1, help="move X-Plane to this workspace (0-based) so you never see it; -1 = leave")
    ap.add_argument("--daystart", type=float, default=8.0)
    ap.add_argument("--dayend", type=float, default=18.0)
    ap.add_argument("--min-px", type=float, default=8.0, help="smallest UAV size (output pixels, longest side)")
    ap.add_argument("--frames", type=int, default=0, help="override frames per model (testing)")
    ap.add_argument("--only", default=None, help="comma list of model numbers, e.g. 1,2,166")
    ap.add_argument("--smoke", action="store_true", help="3 models x 4 frames + 2 groups x 4 into dataset 'uav_smoke'")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--cam-workers", type=int, default=6, help="parallel camstyle processes")
    ap.add_argument("--reference", default="~/Resimler/ornekler")
    ap.add_argument("--camstyle-dir", default="~/Belgeler/GitHub/rpi-cam-style")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    if a.smoke:
        a.name, a.frames, a.only, a.no_post, a.group_frames = "uav_smoke", 4, a.only or "1,60,166", True, 4
    a.out_size = gtd.parse_size(a.out_size)
    a.region = gtd.parse_region(a.region)
    a.hfov, a.vfov = camera_fov(a.focal_mm)
    return a


def main():
    args = parse_args()
    root, csv_path, prog_path, prog = prepare_dataset(args)
    log = Log(os.path.join(root, "fleet_run.log"))
    log(f"başlangıç: {vars(args)}")

    if args.location in c.REGION_OPTIONS:
        c.REGION_OPTIONS[args.location][2] = 0.0     # positions below are meters above sea level
    gtd.args = types.SimpleNamespace(location=args.location, own_p_max=30.0, own_r_max=45.0,
                                     monitor=args.monitor, region=args.region, out_size=args.out_size)

    names = fc.list_models(args.models_dir)
    if args.only:
        wanted = {int(v) for v in args.only.split(",")}
        names = [n for n in names if int(n.split("_")[1]) in wanted]
    singles = [] if args.only_multi else names
    if args.cities == "none":
        cities = []
    elif args.cities == "all":
        cities = list(fc.CITIES)
    else:
        cities = [v.strip() for v in args.cities.split(",")]
        bad = [v for v in cities if v not in fc.CITIES]
        if bad:
            raise SystemExit(f"bilinmeyen şehir: {bad}; seçenekler: {', '.join(fc.CITIES)}")
    bgs = [f"bg_{i:03d}" for i in range(args.bg_units)]
    if args.smoke:
        bgs = bgs[:1]
    groups = [] if args.no_multi else fc.group_plan(names, args.seed)
    if args.smoke:
        groups = [next(g for g in groups if g[0].startswith("m2")), next(g for g in groups if g[0].startswith("m3"))]
    if groups and max(len(m) for _, m in groups) > args.ai_slots:
        raise SystemExit(f"gruplar {max(len(m) for _, m in groups)} UAV istiyor, --ai-slots {args.ai_slots}")
    total = sum(args.frames or fc.frames_for(n) for n in singles) + (len(groups) + len(bgs)) * args.group_frames
    log(f"{len(singles)} model + {len(groups)} çoklu grup + {len(bgs)} arka plan grubu, toplam {total} kare; "
        f"tahmini süre {total * 1.2 / 3600:.1f} sa")

    client = XPlaneConnect()
    client.socket.settimeout(10)
    try:
        client.getDREF("sim/version/xplane_internal_version")
    except Exception as e:
        raise SystemExit(f"X-Plane'e bağlanılamadı (uçuş yüklü mü, XPlaneConnect açık mı?): {e}")
    client.sendVIEW(85)
    client.sendDREF("sim/graphics/view/field_of_view_deg", args.xp_hfov)
    time.sleep(1.0)
    meas, _ = fov_from_projection(client.getDREF("sim/graphics/view/projection_matrix_3d"))
    if abs(meas - args.xp_hfov) > 0.1:
        raise SystemExit(f"FOV tutmadı: hedef {args.xp_hfov}, ölçülen {meas:.3f}")
    client.sendDREF("sim/weather/use_real_weather_bool", 0)
    client.pauseSim(True)
    client.sendDREF("sim/operation/override/override_joystick", 1)
    subprocess.run(["xset", "s", "off", "-dpms", "s", "noblank"], check=False)

    screen = mss.mss()
    win_id = xplane_window_id()
    if win_id is None:
        raise SystemExit("X-Plane penceresi bulunamadı (wmctrl -l)")
    if args.workspace >= 0:
        subprocess.run(["wmctrl", "-i", "-r", win_id, "-t", str(args.workspace)], check=False)
        time.sleep(1.5)
        log(f"X-Plane {args.workspace + 1}. çalışma alanına taşındı; pencere içeriği doğrudan alınıyor")
    frame = grab_window(win_id)
    sh, sw, _ = frame.shape
    if args.out_size and abs((sw / sh) - (args.out_size[0] / args.out_size[1])) > 0.02:
        raise SystemExit(f"X-Plane penceresi {sw}x{sh}, 16:9 değil")

    ctx = types.SimpleNamespace(args=args, client=client, log=log, rng=np.random.default_rng(args.seed),
                                switcher=Switcher(args.xp_dir), root=root, csv_path=csv_path, screen=screen,
                                sw=sw, sh=sh, hfov_meas=round(meas, 3), total_saved=0, win_id=win_id,
                                labels_removed=0, last_check=time.time(), recent=[],
                                period=0.05, settle=6, good=0, crop=crop_mask(args, sw, sh),
                                spots=None, loaded={}, terrain=None, airports=[], city=None)
    log(f"ekran {sw}x{sh}, render FOV {meas:.2f} derece")

    infos = {}

    def info_of(name):
        if name not in infos:
            infos[name] = fc.load_model(args.models_dir, name)
        return infos[name]

    def phantom():
        """A model for the unseen UAV of a background frame, drawn like the UAV frames (talon weighted)."""
        w = np.array([float(args.frames or fc.frames_for(m)) for m in names])
        return info_of(names[int(ctx.rng.choice(len(names), p=w / w.sum()))])

    units = fc.plan_units(singles, groups, bgs, args.frames, args.group_frames, args.seed, cities)
    if "valid" not in prog:                           # fixed at the first start, the same on every resume
        prog["valid"] = sorted(fc.valid_units(units, args.seed))
        json.dump(prog, open(prog_path, "w"), indent=1)
    valid = set(prog["valid"])
    log(f"valid: {len(valid)} birim, {sum(u[3] for u in units if u[0] in valid)} kare "
        f"(birimin tüm kareleri tek tarafta)")
    for key, prefix, models, n, city in units:
        if key in prog["done"]:
            continue
        if cities:
            enter_city(ctx, city)
        for attempt in (1, 2):
            try:
                split = "valid" if key in valid else "train"
                if models:
                    run_unit(ctx, key, prefix, [info_of(m) for m in models], n, split)
                else:
                    run_background(ctx, key, n, split, phantom)
                prog["done"].append(key)
                break
            except RuntimeError as e:
                log(f"[{key}] HATA (deneme {attempt}): {e}")
                remove_model_files(root, prefix)
                ctx.loaded.clear()                    # reload the models on the next try
                if attempt == 2:
                    prog["failed"].append(key)
        json.dump(prog, open(prog_path, "w"), indent=1)

    log(f"model tamam: {len(prog['done'])}, başarısız: {prog['failed']}")
    if not args.no_post:
        post_process(args, log)


if __name__ == "__main__":
    main()
