# Data Generation

## Contents
* **[constants.py](./constants.py):** Contains constants for locations and aircraft used in data generation and labeling calculations
* **[generate_traffic_data.py](./generate_traffic_data.py):** Main file containing code for generating images using X-Plane
* **[generate.sh](./generate.sh) and [generate.ps1](./generate.ps1):** Script files that can be used for running multiple commands in a row. Includes example data generation commands
* **[helpers.py](./helpers.py):** Contains helper functions for data generation
* **[label_traffic_data.py](./label_traffic_data.py):** Main code file for labeling the X-Plane images in YOLO format

## Instructions
Instructions for using the data generation portion of the repository. This will walk you through how to generate images rapidly of different scenes in X-Plane using XPlaneConnect for vision-based aircraft detect-and-avoid algorithm testing. All images and labels (in YOLO format) are saved to the [datasets folder](../../datasets/).

### Setup X-Plane
1. Follow the setup instructions in the [root README](../../README.md).
2. Open X-Plane in full screen mode. If using Windows OS, follow troubleshooting tip #1 below. 
3. From the main menu, select "New Flight"
4. **Setup aircraft**
    * Select any aircraft type for the ownship. Testing was predominantly done with Cessna Skyhawk as ownship.
    * In the upper right corner of the Aircraft menu, select "AI Aircraft." 
    * If there are intruder aircraft already listed, delete all but 1 and then click on "Edit." If there are none, select "Add Aircraft" in the bottom left corner and then select "Edit" for the aircraft that newly appears.
    * In the Edit menu, select one of the following aircraft: Boeing 737-800, Cessna Skyhawk, or King Air C90. Then click "Done." 
5. In the bottom right corner, select "Start Flight" and wait for the scene to appear. 

### Generate Images

*Overview*

All commands to generate data will have the general form of `python3 -m generate_traffic_data -ac [AIRCRAFT NAME]` run from the [data_generation directory](../data_generation/). The default command will generate 5 training and 5 validation images with 0 cloud cover in Palo Alto with the aircraft as currently set. Specifying the aircraft name in the command is for calculation purposes and will NOT change the aircraft in X-Plane. (This must be done manually as specified above). 

The sample command in the `generate` script files is of the form: `python3 -m generate_traffic_data -aw -ac "Cessna Skyhawk" --newac --name "test"`. As specified by the `-aw` flag, the command will generate 10 images (5 training and 5 validation) in each of 6 levels of cloud cover in the default location (Palo Alto) As specified by `--newac`, a prompt will appear in the terminal before data is generated asking the user to confirm that the intruder aircraft type in X-Plane matches the one specified in the command. As specified by `--name "test"`, the data will be generated in a folder called "test". 

Use the `-h` flag to see the different arguments and options available for setting the dataset, flight, and environment specifications. The only required argument is `-ac`/`--craft`, which must be followed by the name of the aircraft type. 

*General Recommendations*

The `-ac` argument is required, and it is recommended that you use `--newac` every time the aircraft needs to be changed. The `--newac` flag will ask the user to verify that the intruder aircraft in X-Plane matches the one specified in the command. The options for aircraft type and location strings can be found in the help text outputed when the `-h` flag is used. Location and weather can be changed as desired in sequential commands, and there will be a longer pause between sets of images generated to allow X-Plane time to load the new environment. `-aw` can be used to specify that the same command should be run for each of the 6 weather types. The number of training and validation images can be specified using the `-nt` and `-nv` arguments. If a `--name` is not specified for the dataset folder, a folder will be generated with the prefix "data_" followed by the current time as a floating point number.

*Instructions*
1. Open a terminal window from this repository and cd into the [data_generation](../data_generation/) directory. 
2. Determine what command(s) you would like to run. If there are multiple commands, you can put them in a script file (such as '.sh' or '.ps1' files).
3. From the [data_generation](../data_generation/) directory, run the command or script file as desired. If `--newac` is used, you must reply 'y' in the terminal when requested. 
4. After responding to that prompt, toggle to the X-Plane window and wait for the images to generate. If the scene does not begin changing rapidly after 15 seconds, return to the terminal window, as an error may have occurred. 
5. Wait for the images to finish generated, which will be evident when the scene stops changing at the same rapid cadence and when there is more than a 15 second pause (which would be to be expected between location switches).

### Camera-Matched Generation (Raspberry Pi HQ camera, IMX477)
Use these flags so the dataset looks like frames from the real camera:
* `--out-size 1280x720`: saves images at camera resolution. Labels are computed from the raw X-Plane size and stay correct.
* `--focal-mm <lens focal length>`: sets `--hfov/--vfov` from the IMX477 sensor (16:9 crop). Without it, the defaults are hfov=50, vfov=40.
* `--monitor <n>`: mss monitor index of the X-Plane screen (1 = primary, the default).
* `--region x,y,w,h`: crop to the X-Plane window if X-Plane runs windowed.

Example (Linux, X-Plane fullscreen on primary monitor, 4 mm lens, 5 train + 5 valid per weather type):
```
python3 -m generate_traffic_data -ac "Cessna Skyhawk" -aw -nt 50 -nv 10 --out-size 1280x720 --focal-mm 4.0 --name rpi_cam_v1
```
Each generated row stores `raw_w,raw_h` in `state_data.csv`. Datasets generated before this change have no such columns and still label correctly.

Set the X-Plane view and window to a 16:9 aspect ratio, otherwise the projection does not match the saved frame.

### Visualize Results
1. Open the [DatasetVisualization.ipynb](../notebooks/) jupyter notebook located in the "notebooks" directory. 
2. Adjust the value of `DATASET_NAME` to match the name of the folder in which the data is stored.
3. Run all cells and view results in the "Check Results of label_traffic_data" section.

### Troubleshooting

1. **If screenshots are all of the same frame even though the X-Plane simulator is updating positions properly:** Configure windowed simulator on X-Plane by navigating to `Flight -> Flight Configuration -> Settings icon in upper right -> Graphics -> Monitor usage -> toggle to “Windowed Simulator”`
2. **If screenshots on MacOS are not capturing X-Plane:** Go to System Preferences > Security & Privacy > Privacy > Screen Recording. Then make sure your IDE and/or Terminal are listed as allowed to take screen recordings on your device. 

### 12 mm IMX477 / Shadow Twin Boom flow
Generates a dataset whose geometry matches a 12 mm lens on the IMX477 (hfov 29.36, vfov 16.76 deg at 16:9, 1280x720).

Requirements:
* XPlaneConnect built from source (the prebuilt Linux `lin.xpl` can lack `CameraCallbacks`). Build target `xpc64` in `xpcPlugin/`, copy to `Resources/plugins/XPlaneConnect/64/lin.xpl`.
* Ownship: Cessna Skyhawk. AI aircraft: `Extra Aircraft/shadow_twin_boom`.

Steps (run from this directory):
1. `python3 check_xplane.py --hfov 29.36` checks the XPC link, FOV write/readback and that screen capture is fresh.
2. `python3 -m generate_traffic_data -ac "Shadow Twin Boom" --focal-mm 12 --xp-hfov 34 --out-size 1280x720 -nt 20 -nv 5 --name probe --newac`
3. `python3 qa_dataset.py probe --n 30` writes `/tmp/xp_qa/overlay_grid.jpg` and prints box size statistics. Check that boxes enclose the aircraft.
4. `./generate_12mm.sh` produces `xp12mm_v1` (train/valid) and `xp12mm_test` (unseen location).

Labels: the box is not computed from the old `daw / distance` constant. The 8 corners of the aircraft 3D box (`constants.AIRCRAFT_BOX`) are projected with the live X-Plane `world_matrix` and `projection_matrix_3d`, then clipped to the frame. Frames where less than half of the box is visible are skipped and resampled. `--xp-hfov` is the FOV X-Plane renders with (default 34); `--focal-mm` sets the angle range where the intruder is placed. Post-process with `rpi-cam-style` (`--focal-mm 12 --source-hfov 34`) to crop to the real camera FOV.

Tests: `cd ../ && ../.venv/bin/python -m pytest data_generation/tests -q`.

## 166 UAV overnight run

`generate_fleet.py` runs every model in `plane_models/` (60 frames each, 180 for `uav_166_tactical_talon`), switches
the AI aircraft with the `UavSwitch` X-Plane plugin (`src/xplane_plugin/uav_switch`), labels with the real mesh
(convex hull of the OBJ vertices projected with the live camera matrices), and converts to the camera look with
`rpi-cam-style` at the end. Steps and checks: `docs/FLEET_GECE.md`.

## 166 UAV fleet dataset (uav_fleet_v3)
`./run_fleet_night.sh uav_fleet_v3` archives old datasets into `datasets/arsiv/` (`archive_datasets.sh`), then renders 120 frames per
model (100 train + 20 valid) and 1000 for `uav_166_tactical_talon` (833 + 167): 20 800 frames. It resumes from `progress.json`.
* **[generate_fleet.py](./generate_fleet.py):** Frame loop. The box is the pixel difference inside the UAV silhouette. Runs a self check every 15 minutes (`<dataset>/kontrol/son_kontrol.jpg`)
* **[overlay_clean.py](./overlay_clean.py):** Finds and removes X-Plane's `<Anchor deployed>` tag
* **[check_run.py](./check_run.py):** Run status for the 15 minute check
* **[draw_boxes.py](./draw_boxes.py):** Red 1 px box overlays

Details (Turkish): [docs/FLEET_GECE.md](../../docs/FLEET_GECE.md).
