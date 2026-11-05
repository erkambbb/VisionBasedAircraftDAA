"""Dry run of run_model() with a fake X-Plane (no simulator needed)."""
import math
import os
import types

import numpy as np
import pytest

import data_generation.constants as c
import data_generation.generate_fleet as gf
import data_generation.generate_traffic_data as gtd
from data_generation import fleet_core as fc


class FakeClient:
    def __init__(self):
        self.sent = []

    def sendDREF(self, k, v): self.sent.append((k, v))
    def sendPOSI(self, v, ac=0): pass

    def getDREFs(self, ks): return [self.getDREF(k) for k in ks]

    def getDREF(self, k):
        if k.endswith("world_matrix"):
            return list(np.eye(4).ravel())
        if k.endswith("projection_matrix_3d"):
            f = 1.0 / math.tan(math.radians(34.0) / 2)
            m = np.zeros((4, 4)); m[0, 0] = f; m[1, 1] = f * 16 / 9; m[2, 2] = -1.0; m[2, 3] = -0.2; m[3, 2] = -1.0
            return list(m.T.ravel())
        if k.endswith("plane1_z"): return [-60.0]
        return [0.0]


class FakeSwitcher:
    def load(self, acf, timeout=30): return acf


def test_run_model_writes_dataset(tmp_path, monkeypatch):
    import cv2
    monkeypatch.setattr(gf, "LOAD_WAIT_S", 0)
    monkeypatch.setattr(gf.time, "sleep", lambda s: None)
    bg = np.random.default_rng(0).integers(150, 160, (1080, 1920, 3), dtype=np.uint8)
    tag = cv2.imread(os.path.join(os.path.dirname(gf.overlay_clean.__file__), "assets", "anchor_label_ref.png"))
    tag = cv2.resize(tag, (153, 39), interpolation=cv2.INTER_CUBIC)   # raw frames are 1.5x the saved size
    calls = {"n": 0}

    def fake_capture(s, cl):                      # odd call = frame with UAV (+ tag), even = empty frame
        calls["n"] += 1
        f = bg.copy()
        if calls["n"] % 2 == 1:
            f[520:560, 900:1020] = 30             # UAV
            f[300:339, 700:853] = tag             # X-Plane tag next to it (must be removed)
        return f

    monkeypatch.setattr(gtd, "capture_frame", fake_capture)
    geo = (960.0, 540.0, 120.0, 40.0)
    poly = np.array([[900, 520], [1020, 520], [1020, 560], [900, 560]], np.float32)
    monkeypatch.setattr(gf, "live_geom", lambda client, sw, sh, hull: (geo, poly, 0.03))
    monkeypatch.setattr(gf, "xplane_active", lambda: True)   # real one sees no X-Plane; wait_front would spin
    gtd.args = types.SimpleNamespace(location="Palo Alto", own_p_max=30.0, own_r_max=45.0, monitor=1, region=None, out_size=(1280, 720))
    lat, lon, _ = c.REGION_OPTIONS["Palo Alto"]
    monkeypatch.setitem(c.REGION_OPTIONS, "Palo Alto", [lat, lon, 0.0])

    args = gf.parse_args(["--name", "t", "--frames", "12"])
    root = tmp_path / "t"
    for sub in ("images/train", "images/valid", "labels/train", "labels/valid"):
        (root / sub).mkdir(parents=True)
    csv = root / "state_data.csv"; csv.write_text(gf.CSV_HEADER)
    hull = np.array([[x, y, z] for x in (-1, 1) for y in (-.2, .5) for z in (-1, 1.4)])
    info = fc.ModelInfo("uav_001_x", "Aircraft/x.acf", hull, radius=1.7, extent=2.4)
    ctx = types.SimpleNamespace(args=args, client=FakeClient(), log=lambda m: None, rng=np.random.default_rng(1),
                                switcher=FakeSwitcher(), root=str(root), csv_path=str(csv), screen=None,
                                sw=1920, sh=1080, hfov_meas=34.0, total_saved=0,
                                labels_removed=0, last_check=0.0, recent=[], period=0.05, settle=6, good=0, crop=None)
    gf.run_model(ctx, 1, "uav_001_x", info)

    assert len(os.listdir(root / "images/train")) == 10 and len(os.listdir(root / "images/valid")) == 2
    assert len(os.listdir(root / "labels/train")) == 10
    lines = csv.read_text().strip().splitlines()
    assert len(lines) == 13 and lines[0].count(",") == lines[1].count(",")
    lab = (root / "labels/train/u001_000.txt").read_text().split()
    assert lab[0] == "0" and all(0 <= float(v) <= 1 for v in lab[1:])
    cx, cy, w, h = (float(v) for v in lab[1:])
    # box = the UAV only (x900-1020 y520-560 of 1920x1080), not stretched towards the tag
    assert abs(cx - 960 / 1920) < 0.003 and abs(cy - 540 / 1080) < 0.004
    assert abs(w - 120 / 1920) < 0.005 and abs(h - 40 / 1080) < 0.006
    assert ctx.labels_removed == 12
    img = cv2.imread(str(root / "images/train/u001_000.jpg"))
    assert not gf.overlay_clean.label_left(img)
    assert any(k == "sim/weather/cloud_type[0]" for k, _ in ctx.client.sent)
    gf.selfcheck(ctx)                              # must pass on clean frames
    assert (root / "kontrol" / "son_kontrol.jpg").exists()
    hits = iter([True] + [False] * 11)                # one doubtful frame only warns
    monkeypatch.setattr(gf.overlay_clean, "label_left", lambda img: next(hits))
    gf.selfcheck(ctx)
    monkeypatch.setattr(gf.overlay_clean, "label_left", lambda img: True)
    with pytest.raises(SystemExit):                   # tags in many frames: removal is broken, stop
        gf.selfcheck(ctx)


def test_diff_bbox_finds_shifted_object_and_ignores_far_shadow():
    rng = np.random.default_rng(0)
    b = rng.integers(90, 100, (1080, 1920, 3), dtype=np.uint8)
    a = b.copy()
    a[500:540, 900:1000] = 20            # UAV, true box x900-1000 y500-540
    a[900:930, 300:420] = 40             # ground shadow far away
    roi = (900.0, 480.0, 100.0, 40.0)    # geometry box shifted left/up of the truth
    cx, cy, w, h = gf.diff_bbox(a, b, roi, 1920, 1080)
    assert abs(cx - 950) < 2 and abs(cy - 520) < 2 and abs(w - 100) < 3 and abs(h - 40) < 3
    assert gf.diff_bbox(b, b, roi, 1920, 1080) is None


def test_edges_agree_rejects_wide_or_cut_boxes():
    geo = (500.0, 300.0, 200.0, 50.0)
    ok = lambda box, g=geo: gf.edges_agree(box, g, 1920, 1080)
    assert ok((501.0, 300.5, 202.0, 51.0))
    assert not ok((520.0, 300.0, 240.0, 50.0))                      # right edge 40 px out: shadow or tag
    assert not ok((480.0, 300.0, 160.0, 50.0))                      # right part of the UAV missed
    assert not ok((500.0, 306.0, 200.0, 62.0))                      # tolerance follows each axis
    assert ok((100.0, 100.0, 12.0, 8.0), (101.0, 100.0, 14.0, 8.0))


def test_edges_agree_on_uav_cut_by_the_frame():
    geo = (50.0, 300.0, 100.0, 80.0)                                # left side cut by the frame
    ok = lambda box: gf.edges_agree(box, geo, 1920, 1080)
    assert ok((50.0, 310.0, 100.0, 40.0))                           # visible part is shorter: fine
    assert not ok((60.0, 300.0, 80.0, 80.0))                        # pixels do not reach the cut edge
    assert not ok((50.0, 300.0, 100.0, 120.0))                      # taller than the UAV: shadow


def test_visible_box_is_tight_on_the_part_inside_the_picture():
    b = np.full((400, 800, 3), 120, np.uint8)
    a = b.copy()
    a[200:210, 100:500] = 20                                        # fuselage
    a[100:200, 100:130] = 20                                        # tail fin at the far end
    crop = np.zeros((400, 800), np.uint8)
    crop[:, 200:] = 1                                               # the fin is outside the picture
    mask = np.ones((400, 800), np.uint8)
    full = gf.diff_bbox(a, b, (300.0, 155.0, 400.0, 110.0), 800, 400, mask)
    vis = gf.visible_box(a, b, full, full, mask, crop, 800, 400)
    assert vis == (350.0, 205.0, 300.0, 10.0)                       # not the 110 px height of the whole
    assert gf.visible_box(a, b, full, full, mask, None, 800, 400) == full
    assert gf.visible_box(a, b, full, full, mask, np.ones_like(crop), 800, 400) == full
    crop[:] = 0
    crop[:, 600:] = 1
    assert gf.visible_box(a, b, full, full, mask, crop, 800, 400) is None


def test_crop_mask_matches_camera_picture():
    m = gf.crop_mask(gf.parse_args(["--name", "t"]), 1920, 1080)
    assert m[540, 960] and not m[0, 0] and not m[540, 5] and not m[5, 960]
    width = int(m[540].sum())
    assert 1600 < width < 1720                                      # tan(14.7)/tan(17) of 1920 px


def test_poly_mask_grows_silhouette_and_outside_share():
    poly = np.array([[100, 100], [200, 100], [200, 140], [100, 140]], np.float32)
    m = gf.poly_mask(poly, 640, 360, (150.0, 120.0, 100.0, 40.0))
    assert m[120, 150] and m[120, 208] and m[92, 150] and not m[120, 230] and not m[70, 150]
    a = np.zeros((360, 640, 3), np.uint8)
    b = a.copy()
    b[120, 150] = 200                                               # inside the UAV: does not count
    assert gf.outside_share(a, b, m) == 0
    b[300:, :] = 200
    assert gf.outside_share(a, b, m) > 0.1


def test_adapt_slows_on_bad_frames_and_recovers():
    ctx = types.SimpleNamespace(settle=6, good=10, period=0.02)
    gf.adapt(ctx, False)
    assert ctx.settle == 7 and ctx.good == 0
    for _ in range(25):
        gf.adapt(ctx, True)
    assert ctx.settle == 6


def test_shards_cover_every_frame_and_unlabelled_are_dropped(tmp_path):
    src = tmp_path / "s"
    for sub in ("images/train", "images/valid", "labels/train", "labels/valid"):
        (src / sub).mkdir(parents=True)
    for i in range(7):
        (src / f"images/train/u{i}.jpg").write_bytes(b"x")
        (src / f"labels/train/u{i}.txt").write_text("0 .5 .5 .1 .1\n" if i else "")
    shards = gf.make_shards(str(src), str(tmp_path / "t"), 3)
    names = sorted(f for s in shards for f in os.listdir(os.path.join(s, "images/train")))
    assert names == [f"u{i}.jpg" for i in range(7)]
    assert all(os.path.exists(os.path.join(s, "labels/train", f[:-4] + ".txt"))
               for s in shards for f in os.listdir(os.path.join(s, "images/train")))
    assert gf.drop_unlabelled(str(src)) == 1
    assert not (src / "images/train/u0.jpg").exists() and not (src / "labels/train/u0.txt").exists()
