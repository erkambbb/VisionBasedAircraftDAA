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
    assert fc.frames_for("uav_001_micro_re_sparrow") == 90
    assert fc.frames_for("uav_166_tactical_talon") == 1150
    assert 165 * 90 + 1150 == 16000


def test_distance_sampling_ranges_and_mix():
    rng = np.random.default_rng(0)
    ds = np.array([fc.sample_distance(rng, radius=1.0, extent=3.0)[0] for _ in range(6000)])
    assert ds.min() >= 0.50 - 1e-9 and ds.max() <= 300.0 + 1e-9
    assert 0.08 < np.mean(ds < 25) < 0.12
    assert 0.47 < np.mean((ds >= 25) & (ds < 180)) < 0.53
    assert 0.37 < np.mean(ds >= 180) < 0.43


def test_small_model_far_band_shrinks_below_its_limit():
    rng = np.random.default_rng(4)
    lim = fc.far_limit_m(0.2, 8.0)                 # ~61 m
    ds = np.array([fc.sample_distance(rng, radius=0.1, extent=0.2)[0] for _ in range(3000)])
    assert ds.max() <= lim + 1e-9 and 0.08 < np.mean(ds < 25) < 0.12


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
    assert 150 <= min(a) and max(a) <= 1000


def test_weather_shares_and_layers_above_ownship():
    rng = np.random.default_rng(3)
    w = [fc.pick_weather(rng) for _ in range(6000)]
    names = [n for n, _ in w]
    for name, share in zip(fc.WEATHER_PRESETS, fc.WEATHER_PROBS):
        assert abs(names.count(name) / len(names) - share) < 0.025, name
    for name, p in w:
        assert all(f"cloud_type[{i}]" in p for i in range(3))
        assert all(p[f"cloud_base_msl_m[{i}]"] >= 450 for i in range(3) if p[f"cloud_type[{i}]"])
        assert (p["rain_percent"] > 0) == (name in ("yagmur", "firtina") or (name == "karisik" and p["rain_percent"] > 0))
        assert (p["thunderstorm_percent"] > 0) == (name == "firtina")
        if name == "sis":
            assert p["visibility_reported_m"] <= 2000


def test_summer_day_and_clock():
    rng = np.random.default_rng(5)
    d = [fc.sample_day(rng) for _ in range(500)]
    assert min(d) >= 171 and max(d) <= 264
    assert fc.zulu_sec(8 * 3600, "New York") == 12 * 3600      # 08:00 EDT = 12:00 Z
    assert fc.zulu_sec(18 * 3600, "Los Angeles") == 1 * 3600   # 18:00 PDT = 01:00 Z next day
    assert fc.zulu_sec(12 * 3600, "Tokyo") == 3 * 3600         # JST, no summer time
    assert fc.zulu_sec(12 * 3600, "Sydney") == 1 * 3600        # AEDT = UTC+11
    assert fc.city_day("London", 200) == 200
    s = [fc.city_day("Sydney", d) for d in range(171, 265)]
    assert all(d >= 353 or d <= 81 for d in s)                 # southern summer: Dec .. Mar


def test_city_spots_near_new_york_origin():
    spots = fc.spot_offsets("New York")
    assert {s[5] for s in spots} == set(fc.SPOT_CATEGORIES) and len(spots) >= 30
    assert all(math.hypot(s[1], s[2]) < 25000 for s in spots)
    mid = next(s for s in spots if s[0] == "midtown_manhattan")
    assert 2500 < mid[1] < 3000 and 1500 < mid[2] < 1900       # about 2.6 km east, 1.7 km north
    name, e, n, g = fc.sample_spot(np.random.default_rng(0), None, 2500.0)
    assert name == "flat" and g == 0 and abs(e) <= 2500
    assert fc.spot_offsets("Palo Alto") is None


def test_apart_and_pitch():
    assert fc.apart([], 0, 0, 5)
    assert not fc.apart([(0.0, 0.0, 2.0)], 3.0, 0.0, 1.0)
    assert fc.apart([(0.0, 0.0, 2.0)], 6.0, 0.0, 1.0)
    rng = np.random.default_rng(6)
    p = np.array([fc.sample_pitch(rng) for _ in range(4000)])
    assert 0.40 < np.mean(p <= -8.0) < 0.52 and p.min() >= -35.0


def test_tilt_ladder_shares_and_bounds():
    rng = np.random.default_rng(3)
    uav = np.array([fc.uav_attitude(rng) for _ in range(20000)])
    for col in range(2):
        v = np.abs(uav[:, col])
        assert v.max() <= fc.UAV_TILT_MAX
        assert abs(np.mean(v == 0) - 0.50) < 0.02
        for k, (lo, hi) in enumerate([(0, 15), (15, 30), (30, 45), (45, 60)], 1):
            assert abs(np.mean((v > lo) & (v <= hi)) - fc.TILT_PROBS[k]) < 0.02
    own = np.array([fc.sample_pitch(rng, down=False) for _ in range(4000)])
    assert np.abs(own).max() <= fc.OWN_PITCH_MAX and abs(np.mean(own == 0) - 0.50) < 0.03


def test_group_plan_counts_and_resume_stable():
    names = [f"uav_{i:03d}_m" for i in range(1, 166)] + [fc.BIG_MODEL]
    plan = fc.group_plan(names, 7)
    assert plan == fc.group_plan(names, 7)
    two = [m for k, m in plan if k.startswith("m2_")]
    more = [m for k, m in plan if k.startswith("m3_")]
    assert len(two) == 100 and len(more) == 100
    assert all(len(m) == 2 for m in two) and all(len(m) in (3, 4) for m in more)
    assert all(len(set(m)) == len(m) for _, m in plan)
    assert (len(two) + len(more)) * fc.GROUP_FRAMES == 4000
    talon = np.mean([fc.BIG_MODEL in m for _, m in plan])
    assert 0.4 < talon < 0.6


def test_bbox_loose_keeps_big_closeup_but_rejects_edge_sliver():
    sw, sh = 1920, 1080
    ok = np.array([[-3000, -2000], [3000, 2500]], float)          # covers the whole frame
    assert bbox_loose(ok, np.array([True, True]), sw, sh) is not None
    sliver = np.array([[1900, 500], [4000, 560]], float)          # mostly outside, tiny inside
    assert bbox_loose(sliver, np.array([True, True]), sw, sh) is None
    behind = np.array([[0, 0], [100, 100]], float)
    assert bbox_loose(behind, np.array([True, False]), sw, sh) is None


def test_edge_angles_put_uav_on_picture_edge_inside_raw_frame():
    import math
    rng = np.random.default_rng(0)
    for _ in range(300):
        h, v = fc.edge_angles(rng, 1.0, 2.0, 29.36, 16.7, 34.0, 50.0, 2.0)
        assert abs(h) <= 17.0 and abs(v) <= 9.9
        spread = math.degrees(math.atan2(2.0, 50.0))
        assert (h, v)[0] == 1.0 or abs(abs(h) - 14.68) <= spread
        assert (h != 1.0) != (v != 2.0)


def test_unit_plan_gives_every_unit_the_same_mix():
    rng = np.random.default_rng(0)
    for n in (20, 90, 1150):
        p = fc.unit_plan(rng, n, 8, 18)
        w = [x["weather"] for x in p]
        assert [w.count(k) for k in fc.WEATHER_PRESETS] == list(fc.quota(n, fc.WEATHER_PROBS))
        changes = sum(a != b for a, b in zip(w, w[1:]))
        assert changes == len(set(w)) - 1                              # one block per preset
        bands = [x["band"] for x in p]
        assert [bands.count(i) for i in range(3)] == list(fc.quota(n, [b[2] for b in fc.DIST_BANDS]))
        t = np.array([x["local_time"] for x in p]) / 3600
        assert t.min() >= 8 and t.max() <= 18 and np.histogram(t, 10, (8, 18))[0].min() >= n // 10 - 1
        a = np.array([x["alt"] for x in p]) / fc.FT
        assert a.min() >= 150 and a.max() <= 1000
        assert all(171 <= x["day"] <= 264 for x in p)
        for k in fc.TILT_KEYS:
            assert [x[k] for x in p].count(0) == fc.quota(n, fc.TILT_PROBS)[0]


def test_spots_stay_away_from_airports():
    rng = np.random.default_rng(1)
    for city in fc.CITIES:
        spots, ap = fc.spot_offsets(city), fc.airport_offsets(city)
        assert ap and all(len(a) == 3 for a in ap)
        for cat in fc.SPOT_CATEGORIES:
            for _ in range(60):
                name, e, n, g = fc.sample_spot(rng, spots, 0, cat, ap)
                assert fc.far_from_airports(e, n, ap)
                if any(s[5] == cat for s in spots):
                    assert next(s for s in spots if s[0] == name)[5] == cat


def test_cities_have_spots_near_origin():
    assert len(fc.CITIES) >= 15 and set(fc.SPOTS) == set(fc.CITIES)
    for city in fc.CITIES:
        spots = fc.spot_offsets(city)
        assert len(spots) >= 3 and len({s[0] for s in spots}) == len(spots)
        assert all(math.hypot(s[1], s[2]) < 30000 for s in spots), city


def test_plan_units_spreads_cities_evenly():
    names = [f"uav_{i:03d}_x" for i in range(1, 166)] + [fc.BIG_MODEL]
    groups = fc.group_plan(names, 7)
    bgs = [f"bg_{i:03d}" for i in range(100)]
    units = fc.plan_units(names, groups, bgs, 0, 20, 7, list(fc.CITIES))
    assert sum(u[3] for u in units) == 22000 and len({u[0] for u in units}) == len(units)
    per = {}
    for u in units:
        per[u[4]] = per.get(u[4], 0) + u[3]
    assert set(per) == set(fc.CITIES) and max(per.values()) - min(per.values()) <= 30
    order = [u[4] for u in units]
    assert sum(a != b for a, b in zip(order, order[1:])) == len(fc.CITIES) - 1   # one visit per city
    for n in names[:20]:
        cs = [u[4] for u in units if u[2] == [n]]
        assert len(cs) == len(set(cs)) == 3                    # 90 frames: 3 blocks, 3 cities
    assert fc.split_blocks(1150) == [31] * 10 + [30] * 28 and fc.split_blocks(4) == [4]
    one = fc.plan_units(names[:3], [], [], 4, 20, 7, ["Tokyo"])
    assert [u[4] for u in one] == ["Tokyo"] * 3


def test_terrain_grid_interpolates_and_falls_back():
    t = fc.Terrain(0.0, 0.0, 100.0, np.array([[0.0, 10.0], [20.0, np.nan]]))
    assert abs(t.ground(50.0, 0.0) - 5.0) < 0.01
    assert abs(t.ground(50.0, 50.0, fallback=7.0) - 10.0) < 0.01          # unloaded corner skipped
    empty = fc.Terrain(0.0, 0.0, 100.0, np.full((2, 2), np.nan))
    assert abs(empty.ground(50.0, 50.0, fallback=7.0) - 7.0) < 0.01
    far = fc.Terrain(0.0, 0.0, 100.0, np.zeros((2, 2)))
    assert abs(far.ground(10000.0, 0.0) + 7.85) < 0.05                 # earth curvature at 10 km


def test_valid_units_takes_whole_units_per_kind_and_spreads_cities():
    names = [f"uav_{i:03d}_x" for i in range(1, 166)] + [fc.BIG_MODEL]
    groups = fc.group_plan(names, 7)
    bgs = [f"bg_{i:03d}" for i in range(fc.BG_UNITS)]
    units = fc.plan_units(names, groups, bgs, 0, 20, 7, list(fc.CITIES))
    valid = fc.valid_units(units, 7)
    assert valid == fc.valid_units(units, 7)                  # the same on every resume
    kinds = {}
    for u in units:
        k = fc.unit_kind(u)
        n, v = kinds.get(k, (0, 0))
        kinds[k] = (n + 1, v + (u[0] in valid))
    assert set(kinds) == {"single", "talon", "m2", "m3", "bg"}
    for k, (n, v) in kinds.items():
        assert v == round(n / 6), (k, n, v)
    per_model = {}
    for u in units:
        if fc.unit_kind(u) == "single" and u[0] in valid:
            per_model[u[2][0]] = per_model.get(u[2][0], 0) + 1
    assert max(per_model.values()) == 1                       # a 3-block model keeps 2 blocks in train
    assert len({u[4] for u in units if u[0] in valid}) == len(fc.CITIES)
    share = sum(u[3] for u in units if u[0] in valid) / sum(u[3] for u in units)
    assert 0.14 < share < 0.19
    assert len(fc.valid_units(units[:2], 7)) == 1 and fc.valid_units(units[:1], 7) == set()
