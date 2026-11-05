import math
import numpy as np
from data_generation import fleet_core as fc
from data_generation.projection import body_points_ogl, box_corners_ogl, bbox_loose, project_points


def _obj(tmp_path, pts):
    p = tmp_path / "x.obj"
    p.write_text("I\n800\nOBJ\n" + "".join(f"VT {x} {y} {z} 0 1 0 0 0\n" for x, y, z in pts) + "IDX10 0 1 2\n")
    return str(p)


def test_obj_vertices_and_hull(tmp_path):
    cube = [(x, y, z) for x in (-1, 1) for y in (0, 2) for z in (-3, 3)] + [(0, 1, 0)]
    pts = fc.obj_vertices(_obj(tmp_path, cube))
    assert pts.shape == (9, 3)
    h = fc.hull_points(pts)
    assert len(h) == 8 and np.allclose(h.min(0), [-1, 0, -3]) and np.allclose(h.max(0), [1, 2, 3])


def test_body_points_matches_box_corners():
    mn, mx = (-1, -0.2, -1), (1, 0.5, 1.4)
    c1 = box_corners_ogl([10, 20, 30], 40, 5, -12, mn, mx)
    import itertools
    pts = np.array([[(mn, mx)[i][0], (mn, mx)[j][1], (mn, mx)[k][2]] for i, j, k in itertools.product((0, 1), repeat=3)])
    assert np.allclose(c1, body_points_ogl(pts, [10, 20, 30], 40, 5, -12))


def test_frames_and_split():
    assert fc.frames_for("uav_001_micro_re_sparrow") == 120
    assert fc.frames_for("uav_166_tactical_talon") == 1000
    assert fc.split_counts(120) == (100, 20) and fc.split_counts(1000) == (833, 167)


def test_distance_sampling_ranges_and_mix():
    rng = np.random.default_rng(0)
    ds = [fc.sample_distance(rng, radius=1.0, extent=3.0)[0] for _ in range(4000)]
    assert min(ds) >= 0.1 - 1e-9 and max(ds) <= 300.0 + 1e-9
    near = np.mean([d < 5.0 for d in ds])
    assert 0.07 < near < 0.13


def test_small_model_gets_shorter_far_limit():
    assert fc.far_limit_m(3.0, 8.0) == 300.0
    lim = fc.far_limit_m(0.2, 8.0)
    assert 55 < lim < 65                       # 2444 * 0.2 / 8


def test_center_distance_is_surface_plus_radius():
    d, dc = fc.sample_distance(np.random.default_rng(1), radius=2.5, extent=3.0)
    assert math.isclose(dc, d + 2.5)


def test_altitude_in_feet_range():
    rng = np.random.default_rng(2)
    a = [fc.sample_altitude_m(rng) / fc.FT for _ in range(2000)]
    assert 170 <= min(a) and max(a) <= 1000


def test_weather_mostly_clear():
    rng = np.random.default_rng(3)
    w = [fc.pick_weather(rng) for _ in range(5000)]
    assert 0.65 < w.count(0) / len(w) < 0.75 and set(w) <= {0, 1, 2}


def test_bbox_loose_keeps_big_closeup_but_rejects_edge_sliver():
    sw, sh = 1920, 1080
    ok = np.array([[-3000, -2000], [3000, 2500]], float)          # covers the whole frame
    assert bbox_loose(ok, np.array([True, True]), sw, sh) is not None
    sliver = np.array([[1900, 500], [4000, 560]], float)          # mostly outside, tiny inside
    assert bbox_loose(sliver, np.array([True, True]), sw, sh) is None
    behind = np.array([[0, 0], [100, 100]], float)
    assert bbox_loose(behind, np.array([True, False]), sw, sh) is None
