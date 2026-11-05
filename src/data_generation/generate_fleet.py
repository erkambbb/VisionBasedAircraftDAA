"""Multi-UAV dataset run for the 12 mm IMX477 camera.

Every model in plane_models/ gets 120 frames (uav_166_tactical_talon: 1000).
Frames: clear weather (mostly), ownship 170-1000 ft above ground, UAV 0.1 m - 300 m away.

Run from src/data_generation:
    python3 -m generate_fleet --smoke                 # 3 models x 4 frames, quick test
    python3 -m generate_fleet --name uav_fleet_v3     # full run
    python3 -m generate_fleet --name uav_fleet_v3 --resume

Needs: X-Plane running in a flight (ownship Cessna 172SP, 1 AI aircraft in the flight setup),
XPlaneConnect and UavSwitch plugins installed, X-Plane window in front on monitor 1.
"""
import sys
sys.path.append('..')

import argparse
import json
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
              "bb_cx,bb_cy,bb_w,bb_h,out_w_px,out_h_px,geo_dx,geo_dy,hfov_meas\n")
LOAD_WAIT_S = 3.0
CLOUD_BLOCK = 20          # frames per weather state (cloud changes need time to settle)
WEATHER_WAIT_S = 1.0
NET_RETRY_S = 3.0
MIN_CLEAR_M = 25.0        # intruder must stay this high above sea level (ground safety)


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

    def load(self, acf_path, timeout=30.0):
        """Returns the path X-Plane reports as loaded; raises RuntimeError on failure."""
        if not os.path.isabs(acf_path):
            acf_path = os.path.join(self.xp_dir, acf_path)      # plugin gets the full path
        self.n += 1
        rid = str(self.n)
        ack = os.path.join(self.dir, "ack.txt")
        if os.path.exists(ack):
            os.remove(ack)
        tmp = os.path.join(self.dir, "request.tmp")
        with open(tmp, "w") as f:
            f.write(f"{rid}\t{acf_path}\n")
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


# ---------------------------------------------------------------- frame geometry
GEOM_REFS = ([f"sim/multiplayer/position/plane1_{a}" for a in ("x", "y", "z", "psi", "the", "phi")]
             + ["sim/graphics/view/world_matrix", "sim/graphics/view/projection_matrix_3d",
                "sim/operation/misc/frame_rate_period"])


def live_geom(client, sw, sh, hull):
    """(geometric box (cx, cy, w, h) or None, convex polygon of the UAV hull in px, seconds per
    rendered frame). One request: every XPlaneConnect round trip costs a rendered frame."""
    v = client.getDREFs(GEOM_REFS)
    x, y, z, psi, the, phi = (r[0] for r in v[:6])
    pts = body_points_ogl(hull, (x, y, z), psi, the, phi)
    px, front = project_points(pts, v[6], v[7], sw, sh)
    box = bbox_loose(px, front, sw, sh, min_visible=0.01)
    poly = None
    if box is not None:
        poly = cv2.convexHull(px.astype(np.float32)).reshape(-1, 2)
    return box, poly, v[8][0]


def live_bbox(client, sw, sh, hull):
    return live_geom(client, sw, sh, hull)[0]


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


def sample_state(rng, info, args):
    """Ownship + intruder placement, or None if the sample is unusable."""
    alt = fc.sample_altitude_m(rng)
    own = Aircraft(0, rng.uniform(-args.enrange, args.enrange), rng.uniform(-args.enrange, args.enrange),
                   alt, rng.uniform(0, 360), float(np.clip(rng.normal(0, 5), -30, 30)),
                   float(np.clip(rng.normal(0, 10), -45, 45)))
    d_surf, d_cen = fc.sample_distance(rng, info.radius, info.extent, args.min_px)
    vang = rng.uniform(-args.vfov / 2, args.vfov / 2)
    hang = rng.uniform(-args.hfov / 2, args.hfov / 2)
    if rng.random() < fc.EDGE_FRACTION:
        hang, vang = fc.edge_angles(rng, hang, vang, args.hfov, args.vfov, args.xp_hfov, d_cen, info.radius)
    intr = gtd.get_intruder_position(own, d_cen, hang, vang, rng.uniform(0, 360))
    if intr.u < MIN_CLEAR_M:
        return None
    return dict(own=own, intr=intr, vang=vang, hang=hang, d_surf=d_surf, d_cen=d_cen, alt=alt)


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


def remove_model_files(root, idx):
    for sub in ("images/train", "images/valid", "labels/train", "labels/valid"):
        d = os.path.join(root, sub)
        for fn in os.listdir(d):
            if fn.startswith(f"u{idx:03d}_"):
                os.remove(os.path.join(d, fn))


# ---------------------------------------------------------------- periodic self check
def selfcheck(ctx):
    """Every 15 min: re-read the last saved frames, make sure nothing foreign is on screen
    (X-Plane tag), labels sit inside the frame, and drop a montage into <dataset>/kontrol/."""
    ctx.last_check = time.time()
    tiles, problems, tags = [], [], []
    for split, fn, bbox in ctx.recent:
        img = cv2.imread(os.path.join(ctx.root, "images", split, fn + ".jpg"))
        if img is None:
            problems.append(f"{fn}: görüntü okunamadı")
            continue
        h, w = img.shape[:2]
        if overlay_clean.label_left(img):
            tags.append(fn)
        if float(img.std()) < 4.0:
            problems.append(f"{fn}: görüntü düz renk (ekran yanlış olabilir)")
        lab = open(os.path.join(ctx.root, "labels", split, fn + ".txt")).read().split()
        cx, cy, bw, bh = (float(v) for v in lab[1:5])
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


# ---------------------------------------------------------------- one model
def run_model(ctx, idx, name, info):
    args, client, log, rng = ctx.args, ctx.client, ctx.log, ctx.rng
    n = args.frames if args.frames else fc.frames_for(name)
    n_train, n_valid = fc.split_counts(n)
    remove_model_files(ctx.root, idx)

    loaded = ctx.switcher.load(info.acf_path)
    log(f"[{idx:03d}] {name}: yüklendi ({loaded}), {n} kare ({n_train} train + {n_valid} valid), "
        f"yarıçap {info.radius:.2f} m, en büyük boyut {info.extent:.2f} m")
    time.sleep(LOAD_WAIT_S)

    rows, offs, saved, tries, rejects = [], [], 0, 0, {"ground": 0, "bbox": 0, "small": 0, "net": 0, "label": 0, "mismatch": 0, "unstable": 0}
    weather, since_weather = None, CLOUD_BLOCK
    net_fail = 0
    while saved < n:
        tries += 1
        if tries > n * 10:
            raise RuntimeError(f"{name}: {tries} denemede {saved}/{n} kare, çok fazla ret {rejects}")
        st = sample_state(rng, info, args)
        if st is None:
            rejects["ground"] += 1
            continue
        try:
            if since_weather >= CLOUD_BLOCK:
                weather = fc.pick_weather(rng)
                client.sendDREF("sim/weather/cloud_type[0]", weather)
                since_weather = 0
                time.sleep(WEATHER_WAIT_S)
            local_time = rng.uniform(args.daystart * 3600, args.dayend * 3600)
            client.sendDREF("sim/time/zulu_time_sec", local_time + c.TIME_OPTIONS[args.location] * 3600)
            client.sendDREF("sim/time/local_date_days", 0)
            gtd.set_position(client, st["own"])
            gtd.set_position(client, st["intr"])
            settle(ctx)
            bbox, poly, ctx.period = live_geom(client, ctx.sw, ctx.sh, info.hull)
            if bbox is None:
                rejects["bbox"] += 1
                continue
            ow, oh = fc.bbox_out_px(bbox, ctx.sw, ctx.hfov_meas)
            if max(ow, oh) < args.min_px:
                rejects["small"] += 1
                continue
            if not getattr(ctx, 'win_id', None):
                wait_front(log)
            frame = grab_frame(ctx)
            # same camera, UAV moved far away -> pixel difference is the UAV
            gtd.set_position(client, Aircraft(1, 0, 6000, 400, 0, pitch=0, roll=0))
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

        geo = bbox
        mask = poly_mask(poly, ctx.sw, ctx.sh, geo)
        frame, label = overlay_clean.remove_label(frame, empty, mask)
        if label == "overlap":
            rejects["label"] += 1                     # the tag sits on the UAV: cannot be erased cleanly
            continue
        if label == "removed":
            ctx.labels_removed += 1
        if overlay_clean.label_left(frame):
            rejects["label"] += 1
            continue
        if outside_share(frame, empty, mask) > MAX_OUTSIDE:
            rejects["unstable"] += 1                  # scene still changing between the two frames
            adapt(ctx, False)
            continue
        bbox = diff_bbox(frame, empty, geo, ctx.sw, ctx.sh, mask)
        if bbox is None:
            rejects["nodiff"] = rejects.get("nodiff", 0) + 1
            adapt(ctx, False)
            if saved == 0 and rejects["nodiff"] >= 12:
                raise RuntimeError(f"{name}: UAV ekranda görünmüyor (model yüklenmemiş olabilir)")
            continue
        if not edges_agree(bbox, geometric_clipped(geo, ctx.sw, ctx.sh), ctx.sw, ctx.sh):
            rejects["mismatch"] += 1                  # pixels and geometry disagree: do not trust the box
            if saved == 0 and rejects["mismatch"] >= 60:
                raise RuntimeError(f"{name}: piksel kutusu geometriyle hiç uyuşmuyor")
            continue
        ow, oh = fc.bbox_out_px(bbox, ctx.sw, ctx.hfov_meas)
        if max(ow, oh) < args.min_px:
            rejects["small"] += 1
            continue
        geo_dx, geo_dy = bbox[0] - geo[0], bbox[1] - geo[1]
        offs.append((geo_dx, geo_dy))
        bbox = visible_box(frame, empty, bbox, geo, mask, ctx.crop, ctx.sw, ctx.sh)
        if bbox is None or min(fc.bbox_out_px(bbox, ctx.sw, ctx.hfov_meas)) < MIN_VISIBLE_PX:
            rejects["outside"] = rejects.get("outside", 0) + 1     # no UAV in the camera picture
            continue
        ow, oh = fc.bbox_out_px(bbox, ctx.sw, ctx.hfov_meas)

        split = "train" if saved < n_train else "valid"
        fn = f"u{idx:03d}_{saved:03d}"
        cv2.imwrite(os.path.join(ctx.root, "images", split, fn + ".jpg"), gtd.resize_frame(frame, args.out_size))
        with open(os.path.join(ctx.root, "labels", split, fn + ".txt"), "w") as f:
            f.write(yolo_line(*bbox, ctx.sw, ctx.sh))
        o, i = st["own"], st["intr"]
        rows.append(f"{fn}.jpg,{name},{split},{args.location},{weather},{local_time:.1f},{st['alt']:.1f},"
                    f"{o.e:.2f},{o.n:.2f},{o.u:.2f},{o.h:.2f},{o.p:.2f},{o.r:.2f},{st['vang']:.3f},{st['hang']:.3f},"
                    f"{st['d_surf']:.3f},{st['d_cen']:.3f},{i.e:.2f},{i.n:.2f},{i.u:.2f},{i.h:.2f},{i.p:.2f},{i.r:.2f},"
                    f"{ctx.sw},{ctx.sh},{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f},{ow:.1f},{oh:.1f},{geo_dx:.1f},{geo_dy:.1f},"
                    f"{ctx.hfov_meas}\n")
        ctx.recent = (ctx.recent + [(split, fn, bbox)])[-12:]
        saved += 1
        since_weather += 1
        ctx.total_saved += 1
        adapt(ctx, True)

        if time.time() - ctx.last_check >= CHECK_EVERY_S:
            selfcheck(ctx)

    with open(ctx.csv_path, "a") as f:
        f.writelines(rows)
    mo = np.mean(np.abs(np.array(offs)), axis=0) if offs else (0, 0)
    log(f"[{idx:03d}] {name}: {saved} kare tamam, denemeler {tries}, retler {rejects}, "
        f"geometri-piksel kayma ort. dx={mo[0]:.1f} dy={mo[1]:.1f} px")


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
               "--jitter", "0.5", "--seed", str(i), "--min-visible", "0"]
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
    ap.add_argument("--name", default="uav_fleet_v3")
    ap.add_argument("--models-dir", default=os.path.expanduser("~/Belgeler/ARIKOVANI/plane_models"))
    ap.add_argument("--xp-dir", default=XP_DIR)
    ap.add_argument("--location", default="Palo Alto")
    ap.add_argument("--enrange", type=float, default=2500.0, help="ownship east/north range (m); flat bay area")
    ap.add_argument("--focal-mm", type=float, default=12.0)
    ap.add_argument("--xp-hfov", type=float, default=c.XP_RENDER_HFOV)
    ap.add_argument("--out-size", default="1280x720")
    ap.add_argument("--monitor", type=int, default=1)
    ap.add_argument("--region", default=None)
    ap.add_argument("--workspace", type=int, default=1, help="move X-Plane to this workspace (0-based) so you never see it; -1 = leave")
    ap.add_argument("--daystart", type=float, default=8.0)
    ap.add_argument("--dayend", type=float, default=17.0)
    ap.add_argument("--min-px", type=float, default=8.0, help="smallest UAV size (output pixels, longest side)")
    ap.add_argument("--frames", type=int, default=0, help="override frames per model (testing)")
    ap.add_argument("--only", default=None, help="comma list of model numbers, e.g. 1,2,166")
    ap.add_argument("--smoke", action="store_true", help="3 models x 4 frames into dataset 'uav_smoke'")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--cam-workers", type=int, default=6, help="parallel camstyle processes")
    ap.add_argument("--reference", default="~/Resimler/ornekler")
    ap.add_argument("--camstyle-dir", default="~/Belgeler/GitHub/rpi-cam-style")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    if a.smoke:
        a.name, a.frames, a.only, a.no_post = "uav_smoke", 4, a.only or "1,60,166", True
    a.out_size = gtd.parse_size(a.out_size)
    a.region = gtd.parse_region(a.region)
    a.hfov, a.vfov = camera_fov(a.focal_mm)
    return a


def main():
    args = parse_args()
    root, csv_path, prog_path, prog = prepare_dataset(args)
    log = Log(os.path.join(root, "fleet_run.log"))
    log(f"başlangıç: {vars(args)}")

    c.REGION_OPTIONS[args.location][2] = 0.0         # positions below are meters above sea level
    gtd.args = types.SimpleNamespace(location=args.location, own_p_max=30.0, own_r_max=45.0,
                                     monitor=args.monitor, region=args.region, out_size=args.out_size)

    names = fc.list_models(args.models_dir)
    if args.only:
        wanted = {int(v) for v in args.only.split(",")}
        names = [n for n in names if int(n.split("_")[1]) in wanted]
    total = sum(args.frames or fc.frames_for(n) for n in names)
    log(f"{len(names)} model, toplam {total} kare; tahmini süre {total * 1.0 / 3600:.1f} sa")

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
    for k, v in (("cloud_type[1]", 0), ("cloud_type[2]", 0), ("cloud_base_msl_m[0]", 4572), ("cloud_tops_msl_m[0]", 5182)):
        client.sendDREF("sim/weather/" + k, v)
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
                                period=0.05, settle=6, good=0, crop=crop_mask(args, sw, sh))
    log(f"ekran {sw}x{sh}, render FOV {meas:.2f} derece")

    for name in names:
        idx = int(name.split("_")[1])
        if name in prog["done"]:
            continue
        info = fc.load_model(args.models_dir, name)
        for attempt in (1, 2):
            try:
                run_model(ctx, idx, name, info)
                prog["done"].append(name)
                break
            except RuntimeError as e:
                log(f"[{idx:03d}] {name}: HATA (deneme {attempt}): {e}")
                remove_model_files(root, idx)
                if attempt == 2:
                    prog["failed"].append(name)
        json.dump(prog, open(prog_path, "w"), indent=1)

    log(f"model tamam: {len(prog['done'])}, başarısız: {prog['failed']}")
    if not args.no_post:
        post_process(args, log)


if __name__ == "__main__":
    main()
