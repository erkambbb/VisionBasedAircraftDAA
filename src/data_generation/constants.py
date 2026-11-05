# Seconds to pause
PAUSE_1 = 10 # before data generation
PAUSE_2 = 0.3 # between each sample

PATH = "../../datasets/"

# Latitude, Longitiude, and Altitude of different region locations
# about 3000m msl
REGION_OPTIONS = {
    "Palo Alto": [37.46358871459961, -122.11750030517578, 3048.0],
    "Boston": [42.37415695190430, -71.01775360107422, 3048.0],
    "Osh Kosh": [43.98562240600586, -88.55644226074219, 3048.0],
    "Reno Tahoe": [39.51367568969727, -119.76921844482422, 3048.0]
}

# shifted 500m north and 500m east
REGION_OPTIONS_SHIFTED = {
    "Palo Alto": [37.46809387207031, -122.11184692382812, 3048.0],
    "Boston": [42.37865447998047, -71.01168823242188, 3048.0],
    "Osh Kosh": [43.99011993408203, -88.55020904541016, 3048.0],
    "Reno Tahoe": [39.51817703247070, -119.76340484619141, 3048.0],
}

# Hours behind Zulu (UTC) time for each region
TIME_OPTIONS = {
    "Palo Alto": 8,
    "Boston": 5,
    "Osh Kosh": 5,
    "Reno Tahoe": 8
}

DAW_OPTIONS = {
    "Cessna Skyhawk": 20000, 
    "Boeing 737-800": 100000, 
    "King Air C90": 40000
}

# (alpha, beta, min)
DIST_PARAMS = {
    "Cessna Skyhawk": (2, 200, 20), 
    "Boeing 737-800": (3, 200, 50), 
    "King Air C90": (2, 200, 20)
}

# --- 12 mm camera / Shadow Twin Boom pipeline ---
# OBJ body-frame bounds (x right wing, y up, z aft; nose = -z), meters
AIRCRAFT_BOX = {
    "Shadow Twin Boom": ((-1.89963, -0.21541, -1.0), (1.89963, 0.52954, 1.40769)),
}
# log-uniform distance range (m) from ownship
DIST_LOGUNIFORM = {"Shadow Twin Boom": (40.0, 1500.0)}
CAM_FOCAL_MM = 12.0
XP_RENDER_HFOV = 34.0
