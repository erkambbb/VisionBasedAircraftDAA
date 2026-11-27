"""Pure helpers for the overnight multi-UAV dataset run (no X-Plane imports)."""
import math
import os
import re
import numpy as np

FLEET_ACF_DIR = "Aircraft/Extra Aircraft/uav_fleet"
CAM_HFOV_DEG = 29.36          # IMX477 + 12 mm, 16:9 crop
OUT_WIDTH = 1280
F_OUT_PX = (OUT_WIDTH / 2) / math.tan(math.radians(CAM_HFOV_DEG) / 2)   # ~2444 px

# Weather presets and their share of the frames. "karisik" is a random mix (layers, haze, light rain).
WEATHER_PRESETS = ("acik", "ince_bulut", "bulutlu", "yagmur", "firtina", "sis", "karisik")
WEATHER_PROBS = (0.40, 0.10, 0.05, 0.05, 0.05, 0.05, 0.30)
ALT_FT = (150.0, 1000.0)                # ownship height above ground
FT = 0.3048
# Distance from the camera to the nearest UAV surface: (low m, high m, share of UAVs, sampling).
DIST_BANDS = ((0.25, 25.0, 0.10, "log"),
              (25.0, 180.0, 0.50, "uniform"),
              (180.0, 300.0, 0.40, "uniform"))
FAR_RANGE_M = (DIST_BANDS[1][0], DIST_BANDS[-1][1])
EDGE_FRACTION = 0.15                    # share of UAVs cut by the picture edge
DOWNLOOK_FRACTION = 0.45                # share of frames looking down onto the city (busy background)
DOWNLOOK_PITCH = (-35.0, -8.0)
# Tilt ladder for pitch and roll: [flat, then 4 quarters of the max angle, random sign]. Used for
# the UAVs (up to UAV_TILT_MAX) and for our own aircraft (dive/climb up to OWN_PITCH_MAX, roll OWN_ROLL_MAX).
TILT_PROBS = (0.50, 0.20, 0.15, 0.10, 0.05)
TILT_KEYS = ("own_p", "own_r", "uav_p", "uav_r")
UAV_TILT_MAX = 60.0
OWN_PITCH_MAX = 10.0
OWN_ROLL_MAX = 45.0
SUMMER_DAYS = (171, 264)                # day of year, 21 June .. 22 September
MAX_UAV = 4                             # AI aircraft slots the run uses (Flight Configuration: 4 AI aircraft)
# Multi-UAV groups: one combination of models, GROUP_FRAMES frames, one weather state.
GROUP_FRAMES = 20
MULTI_GROUPS = {2: 100, 3: 100}         # 2 UAVs: 100 x 20 = 2000 frames; 3-4 UAVs: 2000 frames
MORE_THAN_2 = ((3, 4), (0.6, 0.4))      # UAV count in a "3" group
BG_UNITS = 100                          # background-only frames: 100 x 20 = 2000 (no UAV, empty label)
TALON_IN_GROUP = 0.5                    # share of groups that include uav_166_tactical_talon

# Cities the ownship flies over, all over the world (X-Plane global scenery is installed; the
# X-Plane Landmarks packs add Chicago, Dubai, Las Vegas, London, New York, Sydney, Washington DC).
# name: (origin lat, lon, UTC offset of standard time h, summer time shift h, southern hemisphere)
CITIES = {
    "New York": (40.7400, -74.0150, -5, 1, False),
    "Chicago": (41.8781, -87.6298, -6, 1, False),
    "Washington DC": (38.9072, -77.0369, -5, 1, False),
    "Toronto": (43.6532, -79.3832, -5, 1, False),
    "Miami": (25.7617, -80.1918, -5, 1, False),
    "Las Vegas": (36.1147, -115.1728, -8, 1, False),
    "Los Angeles": (34.0522, -118.2437, -8, 1, False),
    "Seattle": (47.6062, -122.3321, -8, 1, False),
    "London": (51.5074, -0.1278, 0, 1, False),
    "Paris": (48.8566, 2.3522, 1, 1, False),
    "Frankfurt": (50.1109, 8.6821, 1, 1, False),
    "Barcelona": (41.3874, 2.1686, 1, 1, False),
    "Innsbruck": (47.2692, 11.4041, 1, 1, False),
    "Istanbul": (41.0082, 28.9784, 3, 0, False),
    "Dubai": (25.2048, 55.2708, 4, 0, False),
    "Tokyo": (35.6762, 139.6503, 9, 0, False),
    "Sydney": (-33.8688, 151.2093, 10, 1, True),
}
# Where the ownship flies: (name, lat, lon, ground estimate m MSL, spread m, category).
# Categories: "dense" skyscrapers and high-rise blocks, "mid" 4-8 storey blocks, "low" houses and low-rise.
# The ground estimate is only a fallback: the real terrain height comes from the UavSwitch probe grid.
SPOTS = {
    "New York": (
        ("midtown_manhattan", 40.7549, -73.9840, 15.0, 1200.0, "dense"),
        ("lower_manhattan", 40.7075, -74.0113, 5.0, 900.0, "dense"),
        ("hudson_yards_chelsea", 40.7480, -74.0010, 10.0, 900.0, "dense"),
        ("upper_east_side", 40.7736, -73.9566, 20.0, 1000.0, "dense"),
        ("upper_west_side", 40.7870, -73.9754, 25.0, 1000.0, "dense"),
        ("jersey_city_downtown", 40.7178, -74.0431, 5.0, 1200.0, "dense"),
        ("long_island_city", 40.7447, -73.9485, 5.0, 1200.0, "dense"),
        ("downtown_brooklyn", 40.6928, -73.9903, 10.0, 1200.0, "dense"),
        ("fort_lee_cliffside", 40.8510, -73.9701, 80.0, 1200.0, "dense"),
        ("newark_downtown", 40.7357, -74.1724, 25.0, 1200.0, "dense"),
        ("harlem", 40.8116, -73.9465, 20.0, 1200.0, "mid"),
        ("hoboken", 40.7440, -74.0324, 5.0, 900.0, "mid"),
        ("union_city_west_new_york", 40.7795, -74.0238, 50.0, 1200.0, "mid"),
        ("williamsburg", 40.7081, -73.9571, 10.0, 1200.0, "mid"),
        ("bushwick", 40.6944, -73.9213, 20.0, 1200.0, "mid"),
        ("sunset_park", 40.6455, -74.0124, 20.0, 1200.0, "mid"),
        ("south_bronx", 40.8176, -73.9182, 15.0, 1200.0, "mid"),
        ("fordham_bronx", 40.8615, -73.8905, 40.0, 1500.0, "mid"),
        ("astoria", 40.7644, -73.9235, 15.0, 1000.0, "mid"),
        ("park_slope", 40.6710, -73.9814, 30.0, 1200.0, "low"),
        ("flatbush", 40.6409, -73.9590, 15.0, 1500.0, "low"),
        ("bay_ridge", 40.6262, -74.0300, 20.0, 1200.0, "low"),
        ("bensonhurst", 40.6045, -73.9946, 10.0, 1500.0, "low"),
        ("ridgewood", 40.7043, -73.9018, 30.0, 1200.0, "low"),
        ("forest_hills", 40.7196, -73.8448, 30.0, 1200.0, "low"),
        ("riverdale", 40.8900, -73.9120, 60.0, 1200.0, "low"),
        ("staten_island_north", 40.6290, -74.0950, 30.0, 1500.0, "low"),
        ("bayonne", 40.6687, -74.1143, 10.0, 1500.0, "low"),
        ("kearny", 40.7684, -74.1454, 20.0, 1200.0, "low"),
        ("clifton", 40.8584, -74.1638, 50.0, 1500.0, "low"),
        ("montclair", 40.8259, -74.2090, 100.0, 1500.0, "low"),
        ("teaneck", 40.8976, -74.0160, 20.0, 1500.0, "low"),
    ),
    "Chicago": (
        ("loop", 41.8820, -87.6278, 180.0, 900.0, "dense"),
        ("river_north", 41.8925, -87.6340, 180.0, 800.0, "dense"),
        ("streeterville", 41.8920, -87.6200, 180.0, 600.0, "dense"),
        ("south_loop", 41.8580, -87.6260, 180.0, 900.0, "mid"),
        ("lincoln_park", 41.9214, -87.6513, 180.0, 1200.0, "mid"),
        ("wicker_park", 41.9088, -87.6796, 180.0, 1200.0, "mid"),
        ("pilsen", 41.8550, -87.6610, 180.0, 1200.0, "low"),
        ("bridgeport", 41.8380, -87.6510, 180.0, 1200.0, "low"),
        ("oak_park", 41.8850, -87.7845, 190.0, 1500.0, "low"),
    ),
    "Washington DC": (
        ("rosslyn", 38.8960, -77.0720, 30.0, 600.0, "dense"),
        ("ballston", 38.8820, -77.1110, 80.0, 800.0, "dense"),
        ("downtown", 38.9000, -77.0300, 20.0, 900.0, "mid"),
        ("columbia_heights", 38.9300, -77.0300, 60.0, 900.0, "mid"),
        ("bethesda", 38.9847, -77.0947, 90.0, 900.0, "mid"),
        ("silver_spring", 38.9940, -77.0270, 90.0, 900.0, "mid"),
        ("capitol_hill", 38.8880, -76.9980, 20.0, 900.0, "low"),
        ("georgetown", 38.9076, -77.0723, 30.0, 700.0, "low"),
        ("petworth", 38.9420, -77.0250, 70.0, 1000.0, "low"),
    ),
    "Toronto": (
        ("yonge_eglinton", 43.7070, -79.3980, 160.0, 800.0, "dense"),
        ("north_york_centre", 43.7680, -79.4130, 180.0, 800.0, "dense"),
        ("mississauga_centre", 43.5930, -79.6430, 140.0, 900.0, "dense"),
        ("the_annex", 43.6700, -79.4050, 110.0, 800.0, "mid"),
        ("leslieville", 43.6650, -79.3300, 90.0, 900.0, "low"),
        ("scarborough", 43.7730, -79.2570, 170.0, 1500.0, "low"),
        ("etobicoke", 43.6400, -79.5300, 110.0, 1500.0, "low"),
    ),
    "Miami": (
        ("downtown", 25.7740, -80.1930, 3.0, 800.0, "dense"),
        ("brickell", 25.7600, -80.1950, 3.0, 600.0, "dense"),
        ("edgewater", 25.8000, -80.1880, 3.0, 700.0, "dense"),
        ("miami_beach_south", 25.7900, -80.1300, 2.0, 900.0, "dense"),
        ("little_havana", 25.7660, -80.2200, 3.0, 1000.0, "mid"),
        ("coconut_grove", 25.7280, -80.2400, 3.0, 1000.0, "low"),
        ("coral_gables", 25.7490, -80.2640, 4.0, 1200.0, "low"),
        ("north_miami", 25.8900, -80.1860, 3.0, 1200.0, "low"),
    ),
    "Las Vegas": (
        ("downtown_fremont", 36.1700, -115.1420, 610.0, 800.0, "dense"),
        ("north_strip", 36.1400, -115.1620, 620.0, 700.0, "dense"),
        ("arts_district", 36.1580, -115.1530, 615.0, 800.0, "mid"),
        ("summerlin", 36.1700, -115.3000, 830.0, 1500.0, "low"),
        ("spring_valley", 36.1080, -115.2450, 700.0, 1500.0, "low"),
        ("east_las_vegas", 36.1550, -115.1000, 560.0, 1500.0, "low"),
        ("henderson", 36.0400, -115.0500, 560.0, 1500.0, "low"),
    ),
    "Los Angeles": (
        ("downtown", 34.0500, -118.2550, 90.0, 900.0, "dense"),
        ("century_city", 34.0570, -118.4170, 70.0, 600.0, "dense"),
        ("koreatown", 34.0610, -118.3000, 70.0, 1000.0, "mid"),
        ("hollywood", 34.0980, -118.3290, 110.0, 1000.0, "mid"),
        ("pasadena", 34.1478, -118.1445, 260.0, 1000.0, "mid"),
        ("santa_monica", 34.0300, -118.5000, 30.0, 900.0, "mid"),
        ("silver_lake", 34.0870, -118.2700, 150.0, 1000.0, "low"),
        ("east_la", 34.0300, -118.1700, 60.0, 1200.0, "low"),
        ("culver_city", 34.0210, -118.3960, 30.0, 1000.0, "low"),
    ),
    "Seattle": (
        ("downtown", 47.6080, -122.3350, 40.0, 900.0, "dense"),
        ("south_lake_union", 47.6250, -122.3380, 20.0, 700.0, "dense"),
        ("bellevue", 47.6150, -122.1950, 60.0, 900.0, "dense"),
        ("capitol_hill", 47.6230, -122.3120, 100.0, 900.0, "mid"),
        ("ballard", 47.6680, -122.3850, 20.0, 1000.0, "mid"),
        ("fremont", 47.6510, -122.3500, 40.0, 900.0, "low"),
        ("queen_anne", 47.6370, -122.3570, 120.0, 900.0, "low"),
        ("west_seattle", 47.5650, -122.3870, 80.0, 1200.0, "low"),
    ),
    "London": (
        ("city_of_london", 51.5150, -0.0900, 15.0, 700.0, "dense"),
        ("canary_wharf", 51.5050, -0.0200, 5.0, 600.0, "dense"),
        ("westminster", 51.4990, -0.1350, 10.0, 900.0, "mid"),
        ("southwark", 51.5020, -0.0900, 5.0, 700.0, "mid"),
        ("camden", 51.5390, -0.1430, 30.0, 1000.0, "mid"),
        ("croydon", 51.3760, -0.0980, 50.0, 1000.0, "mid"),
        ("islington", 51.5380, -0.1030, 30.0, 1000.0, "low"),
        ("clapham", 51.4620, -0.1380, 20.0, 1200.0, "low"),
        ("hackney", 51.5450, -0.0550, 20.0, 1200.0, "low"),
        ("greenwich", 51.4800, 0.0000, 20.0, 1000.0, "low"),
    ),
    "Paris": (
        ("la_defense", 48.8920, 2.2380, 50.0, 700.0, "dense"),
        ("montparnasse", 48.8420, 2.3220, 60.0, 600.0, "dense"),
        ("olympiades", 48.8260, 2.3660, 50.0, 600.0, "dense"),
        ("centre", 48.8600, 2.3470, 35.0, 1200.0, "mid"),
        ("marais_bastille", 48.8550, 2.3650, 35.0, 900.0, "mid"),
        ("montmartre", 48.8860, 2.3410, 80.0, 900.0, "mid"),
        ("boulogne", 48.8350, 2.2400, 35.0, 1000.0, "mid"),
        ("vincennes", 48.8470, 2.4390, 60.0, 1200.0, "low"),
        ("saint_denis", 48.9360, 2.3570, 35.0, 1200.0, "low"),
        ("creteil", 48.7900, 2.4550, 50.0, 1200.0, "low"),
    ),
    "Frankfurt": (
        ("bankenviertel", 50.1120, 8.6720, 100.0, 600.0, "dense"),
        ("innenstadt", 50.1130, 8.6850, 100.0, 800.0, "mid"),
        ("nordend", 50.1280, 8.6900, 110.0, 900.0, "mid"),
        ("bockenheim", 50.1220, 8.6450, 105.0, 900.0, "mid"),
        ("offenbach", 50.1000, 8.7650, 100.0, 1000.0, "mid"),
        ("sachsenhausen", 50.1000, 8.6850, 110.0, 900.0, "low"),
        ("eschborn", 50.1440, 8.5700, 130.0, 1000.0, "low"),
        ("bad_homburg", 50.2270, 8.6180, 200.0, 1200.0, "low"),
    ),
    "Barcelona": (
        ("diagonal_mar", 41.4100, 2.2170, 5.0, 700.0, "dense"),
        ("eixample", 41.3910, 2.1650, 40.0, 1000.0, "mid"),
        ("gracia", 41.4030, 2.1560, 80.0, 800.0, "mid"),
        ("sants", 41.3750, 2.1350, 40.0, 800.0, "mid"),
        ("hospitalet", 41.3600, 2.1000, 30.0, 900.0, "mid"),
        ("barceloneta", 41.3800, 2.1900, 3.0, 600.0, "mid"),
        ("sarria", 41.4000, 2.1200, 120.0, 900.0, "low"),
        ("badalona", 41.4500, 2.2470, 20.0, 1000.0, "low"),
    ),
    "Innsbruck": (
        ("innsbruck_east", 47.2700, 11.4300, 575.0, 600.0, "mid"),
        ("rum", 47.2860, 11.4580, 590.0, 700.0, "low"),
        ("hall_in_tirol", 47.2830, 11.5080, 575.0, 900.0, "low"),
    ),
    "Istanbul": (
        ("levent_maslak", 41.0820, 29.0110, 80.0, 900.0, "dense"),
        ("sisli_mecidiyekoy", 41.0670, 28.9930, 100.0, 800.0, "dense"),
        ("atasehir", 40.9920, 29.1200, 100.0, 800.0, "dense"),
        ("beyoglu", 41.0340, 28.9770, 60.0, 800.0, "mid"),
        ("besiktas", 41.0430, 29.0050, 40.0, 800.0, "mid"),
        ("fatih", 41.0150, 28.9500, 40.0, 1000.0, "mid"),
        ("kadikoy", 40.9900, 29.0300, 30.0, 900.0, "mid"),
        ("uskudar", 41.0250, 29.0300, 60.0, 1000.0, "low"),
        ("sariyer", 41.1670, 29.0500, 80.0, 1200.0, "low"),
    ),
    "Dubai": (
        ("downtown_burj", 25.1972, 55.2744, 5.0, 800.0, "dense"),
        ("business_bay", 25.1850, 55.2650, 5.0, 700.0, "dense"),
        ("sheikh_zayed_road", 25.2200, 55.2800, 5.0, 900.0, "dense"),
        ("dubai_marina", 25.0800, 55.1400, 5.0, 800.0, "dense"),
        ("jlt", 25.0700, 55.1450, 10.0, 600.0, "dense"),
        ("deira", 25.2700, 55.3100, 5.0, 900.0, "mid"),
        ("al_barsha", 25.1100, 55.2000, 20.0, 1000.0, "mid"),
        ("jumeirah", 25.2000, 55.2400, 5.0, 1200.0, "low"),
        ("al_quoz", 25.1400, 55.2300, 15.0, 1000.0, "low"),
    ),
    "Tokyo": (
        ("shinjuku", 35.6900, 139.6990, 40.0, 800.0, "dense"),
        ("shibuya", 35.6580, 139.7016, 30.0, 700.0, "dense"),
        ("marunouchi", 35.6810, 139.7670, 5.0, 700.0, "dense"),
        ("shinagawa", 35.6280, 139.7390, 5.0, 800.0, "dense"),
        ("ikebukuro", 35.7300, 139.7110, 35.0, 700.0, "dense"),
        ("koto", 35.6720, 139.8170, 2.0, 1000.0, "mid"),
        ("kawasaki", 35.5310, 139.7030, 5.0, 1000.0, "mid"),
        ("setagaya", 35.6460, 139.6530, 40.0, 1500.0, "low"),
        ("nakano", 35.7070, 139.6650, 40.0, 1200.0, "low"),
        ("nerima", 35.7360, 139.6520, 45.0, 1500.0, "low"),
    ),
    "Sydney": (
        ("cbd", -33.8688, 151.2093, 20.0, 800.0, "dense"),
        ("north_sydney", -33.8390, 151.2070, 60.0, 600.0, "dense"),
        ("parramatta", -33.8150, 151.0010, 20.0, 800.0, "dense"),
        ("chatswood", -33.7970, 151.1830, 90.0, 600.0, "dense"),
        ("surry_hills", -33.8850, 151.2110, 40.0, 800.0, "mid"),
        ("bondi", -33.8910, 151.2620, 30.0, 1000.0, "low"),
        ("newtown", -33.8970, 151.1790, 30.0, 900.0, "low"),
        ("manly", -33.7970, 151.2850, 20.0, 900.0, "low"),
        ("strathfield", -33.8790, 151.0830, 30.0, 1200.0, "low"),
    ),
}
SPOT_CATEGORIES = ("dense", "mid", "low")
CATEGORY_PROBS = (0.45, 0.25, 0.30)
# The ownship never flies near an airport, so no aircraft on the ground can show up in a frame.
# (Parked/static aircraft are also off in the X-Plane settings: renopt_static_acf 0.)
# Land airports from X-Plane's apt.dat near each city (make_city_airports.py writes city_airports.py):
# (ident, lat, lon, longest runway m). Big fields keep AIRPORT_CLEAR_M, small strips SMALL_CLEAR_M.
from data_generation.city_airports import AIRPORTS   # noqa: E402
AIRPORT_CLEAR_M = 4000.0
SMALL_CLEAR_M = 1500.0
BIG_RUNWAY_M = 1500.0
EARTH_R = 6371000.0
# make_fleet_acfs.py uses the shadow_twin_boom template: its CG (the position X-Plane reports)
# is 3.3 ft aft and 0.3 ft above the OBJ origin. Body points are taken relative to the CG.
CG_OFFSET_M = (0.0, 0.3 * 0.3048, 3.3 * 0.3048)
BIG_MODEL = "uav_166_tactical_talon"
FRAMES_PER_MODEL, FRAMES_BIG_MODEL = 90, 1150      # single-UAV frames: 165 x 90 + 1150 = 16 000


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


def sample_distance(rng, radius, extent, min_px=8.0, band=None):
    """(d_surface, d_center) in meters. d_center = d_surface + radius (aircraft origin to camera).
    `band` (index into DIST_BANDS) comes from the unit plan; None = random by share.
    A band beyond the far limit of a small model (too few pixels) shrinks to the part below it."""
    lim = far_limit_m(extent, min_px)
    k = int(rng.choice(len(DIST_BANDS), p=[b[2] for b in DIST_BANDS])) if band is None else int(band)
    lo, hi, _, kind = DIST_BANDS[k]
    if lo >= lim:
        lo, hi, _, kind = next(b for b in DIST_BANDS if b[0] < lim <= b[1])
    hi = min(hi, lim)
    if kind == "log":
        d = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    else:
        d = float(rng.uniform(lo, hi))
    return d, d + radius


def angular_radius_deg(radius, d_center):
    return math.degrees(math.atan2(radius, d_center))


def apart(placed, hang, vang, ang_r, margin_deg=1.0):
    """True if a UAV at (hang, vang) with angular radius ang_r does not overlap any placed one
    (list of (hang, vang, ang_r)) in the picture. Overlapping UAVs cannot be boxed separately."""
    return all(math.hypot(hang - h, vang - v) > 1.25 * (ang_r + r) + margin_deg for h, v, r in placed)


def sample_tilt_cat(rng):
    return int(rng.choice(len(TILT_PROBS), p=TILT_PROBS))


def tilt_value(rng, cat, max_deg):
    """Signed tilt (deg). cat 0: flat; cat k (1..4): uniform in the k-th quarter of max_deg, random sign."""
    if cat == 0:
        return 0.0
    v = float(rng.uniform((cat - 1) / 4 * max_deg, cat / 4 * max_deg))
    return v if rng.random() < 0.5 else -v


def uav_attitude(rng, cat_p=None, cat_r=None):
    """(pitch, roll) of a UAV in degrees, both on the tilt ladder up to UAV_TILT_MAX."""
    cat_p = sample_tilt_cat(rng) if cat_p is None else cat_p
    cat_r = sample_tilt_cat(rng) if cat_r is None else cat_r
    return tilt_value(rng, cat_p, UAV_TILT_MAX), tilt_value(rng, cat_r, UAV_TILT_MAX)


def sample_pitch(rng, down=None, cat=None):
    """Camera pitch (deg): often looking down onto the city so the UAV sits on a busy background;
    otherwise a dive or climb of up to OWN_PITCH_MAX on the tilt ladder."""
    if (rng.random() < DOWNLOOK_FRACTION) if down is None else down:
        return float(rng.uniform(*DOWNLOOK_PITCH))
    return tilt_value(rng, sample_tilt_cat(rng) if cat is None else cat, OWN_PITCH_MAX)


def sample_day(rng):
    return int(rng.integers(SUMMER_DAYS[0], SUMMER_DAYS[1] + 1))


def city_day(city, day):
    """Day of year in the city's summer: SUMMER_DAYS are northern; the south is half a year later."""
    return int((day + 182) % 365) if CITIES[city][4] else int(day)


def zulu_sec(local_sec, city):
    """Zulu time of day for a summer local clock time in `city`."""
    _, _, utc_std, dst, _ = CITIES[city]
    return (local_sec - (utc_std + dst) * 3600) % 86400


def region_origin(location):
    if location in CITIES:
        return CITIES[location][:2]
    from data_generation import constants as c
    return tuple(c.REGION_OPTIONS[location][:2])


def enu_offset(location, lat, lon):
    lat0, lon0 = region_origin(location)
    return (lon - lon0) * 111320.0 * math.cos(math.radians(lat0)), (lat - lat0) * 111320.0


def spot_offsets(location):
    """[(name, east m, north m, ground m, spread m, category)] relative to the region origin, or None."""
    if location not in SPOTS:
        return None
    return [(name, *enu_offset(location, lat, lon), ground, spread, cat)
            for name, lat, lon, ground, spread, cat in SPOTS[location]]


def airport_offsets(location):
    """[(east m, north m, clear radius m)] of the airports near the city."""
    return [(*enu_offset(location, lat, lon), AIRPORT_CLEAR_M if rw >= BIG_RUNWAY_M else SMALL_CLEAR_M)
            for _, lat, lon, rw in AIRPORTS.get(location, ())]


def far_from_airports(e, n, airports):
    return all(math.hypot(e - a[0], n - a[1]) >= (a[2] if len(a) > 2 else AIRPORT_CLEAR_M) for a in airports)


def sample_spot(rng, spots, enrange, category=None, airports=()):
    """(name, east, north, ground estimate) of the ownship, at least AIRPORT_CLEAR_M from every airport.
    Without spots: the flat area +-enrange at sea level."""
    if not spots:
        return "flat", float(rng.uniform(-enrange, enrange)), float(rng.uniform(-enrange, enrange)), 0.0
    pool = [s for s in spots if category is None or s[5] == category] or list(spots)
    for _ in range(100):
        name, e, n, ground, spread, _ = pool[int(rng.integers(len(pool)))]
        e, n = e + float(rng.uniform(-spread, spread)), n + float(rng.uniform(-spread, spread))
        if far_from_airports(e, n, airports):
            return name, e, n, ground
    raise RuntimeError(f"{category}: havalimanından uzak nokta bulunamadı")


def quota(n, probs):
    """Counts per share for n items (largest remainder), summing to n."""
    raw = np.asarray(probs, float) / np.sum(probs) * n
    cnt = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - cnt), kind="stable")[:n - cnt.sum()]:
        cnt[i] += 1
    return cnt


def spread_values(rng, n, lo, hi):
    """n values covering [lo, hi] evenly (one per equal slice), in random order."""
    return list(lo + (hi - lo) * (rng.permutation(n) + rng.random(n)) / n)


def unit_plan(rng, n, daystart, dayend):
    """Per-frame conditions for one unit (model, group or background set), so every unit gets the
    same mix: weather (contiguous blocks, shares WEATHER_PROBS), distance band of the first UAV,
    city category, down-looking camera, and clock time and altitude spread over their whole range.
    The frame with index `saved` uses entry `saved`: a rejected frame retries the same conditions."""
    order = rng.permutation(len(WEATHER_PRESETS))
    wc = quota(n, WEATHER_PROBS)
    weather = [WEATHER_PRESETS[i] for i in order for _ in range(wc[i])]

    def shuffled(probs):
        return list(rng.permutation(np.repeat(np.arange(len(probs)), quota(n, probs))))

    band = shuffled([b[2] for b in DIST_BANDS])
    cat = shuffled(CATEGORY_PROBS)
    down = shuffled((1 - DOWNLOOK_FRACTION, DOWNLOOK_FRACTION))
    times = spread_values(rng, n, daystart * 3600, dayend * 3600)
    alts = spread_values(rng, n, ALT_FT[0] * FT, ALT_FT[1] * FT)
    days = spread_values(rng, n, SUMMER_DAYS[0], SUMMER_DAYS[1] + 1)
    tilt = {k: shuffled(TILT_PROBS) for k in TILT_KEYS}
    return [dict(weather=weather[i], band=int(band[i]), category=SPOT_CATEGORIES[cat[i]], down=bool(down[i]),
                 local_time=float(times[i]), alt=float(alts[i]), day=int(days[i]),
                 **{k: int(tilt[k][i]) for k in TILT_KEYS}) for i in range(n)]


BLOCK_FRAMES = 30     # a model's frames are split into blocks of about this size, each in another city


def split_blocks(n, size=BLOCK_FRAMES):
    """Near-equal block sizes summing to n: 90 -> [30, 30, 30], 1150 -> 38 x ~30, 20 -> [20]."""
    k = max(1, round(n / size))
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def assign_cities(counts, seed):
    """One city per block for blocks listed as counts per unit, spread evenly: the cities go round in
    a shuffled order, and the blocks of one unit always get different cities."""
    rng = np.random.default_rng([seed, 77])
    names = list(CITIES)
    order = [names[i] for i in rng.permutation(len(names))]
    out, k = [], 0
    for c in counts:
        out.append([order[(k + j) % len(order)] for j in range(c)])
        k += c
    return out


def plan_units(singles, groups, bgs, frames, group_frames, seed, cities):
    """[(key, file prefix, models, frames, city)] in run order. A model's frames are split into
    blocks of about BLOCK_FRAMES, each block in another city; groups and background sets are one
    block. Cities go round evenly, and the run visits one city at a time (few scenery loads)."""
    blocks = [[(f"u{int(n.split('_')[1]):03d}_{j:02d}", [n], k) for j, k in enumerate(split_blocks(frames or frames_for(n)))]
              for n in singles]
    blocks += [[(key, models, group_frames)] for key, models in groups]
    blocks += [[(key, [], group_frames)] for key in bgs]
    if not cities:
        return [(key, key, models, k, None) for unit in blocks for key, models, k in unit]
    names = list(CITIES)
    picks = assign_cities([len(u) for u in blocks], seed)
    units = [(key, key, models, k, names.index(c) if c in cities else None)
             for unit, cs in zip(blocks, picks) for (key, models, k), c in zip(unit, cs)]
    # cities left out by the caller: their blocks go round the chosen ones
    i = 0
    for j, u in enumerate(units):
        if u[4] is None:
            units[j] = u[:4] + (names.index(cities[i % len(cities)]),)
            i += 1
    units.sort(key=lambda u: u[4])
    return [u[:4] + (names[u[4]],) for u in units]


class Terrain:
    """Ground height (m, in the ENU 'up' of the region origin) on a regular east/north grid, from
    X-Plane terrain probes. Earth curvature is folded in: ENU up drops d^2 / 2R below the ground."""

    def __init__(self, e0, n0, step, msl):
        self.e0, self.n0, self.step, self.msl = e0, n0, step, np.asarray(msl, float)

    @staticmethod
    def grid(spots, margin=800.0, step=400.0):
        e = [s[1] for s in spots]
        n = [s[2] for s in spots]
        r = max(s[4] for s in spots) + margin
        es = np.arange(min(e) - r, max(e) + r + step, step)
        ns = np.arange(min(n) - r, max(n) + r + step, step)
        return es, ns

    def ground(self, e, n, fallback=0.0):
        ny, nx = self.msl.shape
        x = min(max((e - self.e0) / self.step, 0.0), nx - 1.000001)
        y = min(max((n - self.n0) / self.step, 0.0), ny - 1.000001)
        i, j = int(y), int(x)
        q = self.msl[i:i + 2, j:j + 2]
        fy, fx = y - i, x - j
        w = np.array([[(1 - fx) * (1 - fy), fx * (1 - fy)], [(1 - fx) * fy, fx * fy]])
        ok = ~np.isnan(q) & (w > 1e-9)
        h = float((w[ok] * q[ok]).sum() / w[ok].sum()) if ok.any() else fallback   # skip unloaded corners
        return float(h) - (e * e + n * n) / (2 * EARTH_R)


def group_plan(names, seed):
    """[(key, models)] for the multi-UAV groups, the same on every resume."""
    rng = np.random.default_rng([seed, 99])
    others = [n for n in names if n != BIG_MODEL]
    plan = []
    for kind in sorted(MULTI_GROUPS):
        for g in range(MULTI_GROUPS[kind]):
            k = kind if kind == 2 else int(rng.choice(MORE_THAN_2[0], p=MORE_THAN_2[1]))
            k = min(k, len(names))
            if BIG_MODEL in names and rng.random() < TALON_IN_GROUP:
                models = [BIG_MODEL] + list(rng.choice(others, k - 1, replace=False))
            else:
                models = list(rng.choice(others if len(others) >= k else names, k, replace=False))
            plan.append((f"m{kind}_{g:03d}", [str(m) for m in models]))
    return plan


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
    """(preset name, {X-Plane 11 weather dataref: value}). Every layer is always written, so
    nothing from the previous state is left over. Cloud bases stay above the ownship (max ~340 m MSL)."""
    name = str(rng.choice(WEATHER_PRESETS, p=WEATHER_PROBS))
    return name, weather_params(rng, name)


def weather_params(rng, name):
    u = rng.uniform
    p = {"visibility_reported_m": u(25000, 40000), "rain_percent": 0.0, "thunderstorm_percent": 0.0}
    for i in range(3):
        p.update({f"cloud_type[{i}]": 0, f"cloud_coverage[{i}]": 0.0,
                  f"cloud_base_msl_m[{i}]": 4572.0 + 1500 * i, f"cloud_tops_msl_m[{i}]": 5182.0 + 1500 * i})

    def layer(i, kind, base, thick, cover):
        p.update({f"cloud_type[{i}]": int(kind), f"cloud_coverage[{i}]": float(cover),
                  f"cloud_base_msl_m[{i}]": float(base), f"cloud_tops_msl_m[{i}]": float(base + thick)})

    if name == "ince_bulut":
        layer(0, 1, u(6000, 8000), 300, u(1, 3))
    elif name == "bulutlu":
        layer(0, rng.choice((2, 3)), u(900, 2000), u(300, 1000), u(2, 4))
        if rng.random() < 0.5:
            layer(1, 1, u(6000, 8000), 300, u(1, 3))
    elif name == "yagmur":
        layer(0, 4, u(600, 1500), u(1000, 2500), u(5, 6))
        p.update(rain_percent=u(0.3, 0.8), visibility_reported_m=u(3000, 10000))
    elif name == "firtina":
        base = u(500, 1200)
        layer(0, 4, base, u(2000, 4000), 6)
        layer(1, 4, base + 5000, 1500, u(4, 6))
        p.update(rain_percent=u(0.7, 1.0), thunderstorm_percent=u(0.5, 1.0), visibility_reported_m=u(2000, 6000))
    elif name == "sis":
        layer(0, 5, u(450, 800), u(200, 500), u(4, 6))
        p.update(visibility_reported_m=u(500, 2000))
    elif name == "karisik":
        base = u(700, 2500)
        for i in range(int(rng.integers(1, 4))):
            layer(i, rng.integers(1, 6), base, u(200, 1500), u(1, 6))
            base += u(1500, 3000)
        p["visibility_reported_m"] = u(4000, 40000)
        if rng.random() < 0.25:
            p["rain_percent"] = u(0.05, 0.3)
    return p


def bbox_out_px(bbox, sw, xp_hfov):
    """(w, h) of a raw-frame bbox in output-frame pixels (1280 wide, camera FOV)."""
    f_raw = (sw / 2) / math.tan(math.radians(xp_hfov) / 2)
    k = F_OUT_PX / f_raw
    return bbox[2] * k, bbox[3] * k
