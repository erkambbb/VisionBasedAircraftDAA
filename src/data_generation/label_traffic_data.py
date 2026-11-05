import pandas as pd
import numpy as np
import cv2
from data_generation.helpers import Aircraft
import argparse
import json
import os
import data_generation.constants as c

def get_bb_size(o, i, aw0=0, daw=1):
    """Gets height and width of bounding box"""
    
    x = i.n - o.n
    y = -(i.e - o.e) # right-handed coordinates
    z = i.u - o.u
    
    # Get height and width of bounding box
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    w = float(daw) * (1 / r) + aw0
    h = (3 / 8) * w
    return h, w

def yolo_line(cx, cy, w, h, sw, sh):
    """One YOLO label line (class 0) from a pixel box and the raw frame size"""
    return "0 %.6f %.6f %.6f %.6f\n" % (cx / sw, cy / sh, w / sw, h / sh)

def gen_labels(metadata, total_images):
    """Writes bounding box data in Yolo format
    
    Parameters
    ----------
    metadata : dict
        contains metadata for the dataset
    total_images : number of images in dataset
    """

    # Load in the positions
    outdir = metadata['outdir']
    data_file = os.path.join(outdir, 'state_data.csv')
    df = pd.read_csv(data_file)
    # Newer datasets store the raw X-Plane capture size per row (raw_w, raw_h).
    # Labels are normalized with that size, so resizing the saved images
    # (e.g. to the camera resolution) does not change the labels.
    has_raw_size = 'raw_w' in df.columns
    has_bb = 'bb_w' in df.columns
    if not has_raw_size and not has_bb:
        sh, sw, _ = cv2.imread(os.path.join(outdir, "images", "train", "0.jpg")).shape

    curr_dir = os.path.join(outdir, "labels", "train", "")
    for i in range(total_images - metadata['num_train'] - metadata['num_valid'], total_images):
        if i == total_images - metadata['num_valid']: curr_dir = os.path.join(outdir, "labels", "valid", "")
        xp_data = df.iloc[i]
        if has_raw_size:
            sh, sw = int(xp_data['raw_h']), int(xp_data['raw_w'])
        file_name = curr_dir + str(i) + ".txt"
        if has_bb:
            with open(file_name, 'w+') as fd:
                fd.write(yolo_line(xp_data['bb_cx'], xp_data['bb_cy'], xp_data['bb_w'], xp_data['bb_h'], sw, sh))
            continue
        own = Aircraft(0, xp_data['e0'], xp_data['n0'], xp_data['u0'], xp_data['h0'])
        intr = Aircraft(0, xp_data['e1'], xp_data['n1'], xp_data['u1'], xp_data['h1'])
        xp, yp = xp_data['intr_x'], xp_data['intr_y']
        daw = c.DAW_OPTIONS[metadata['ac']]
        h, w = get_bb_size(own, intr, daw=daw)

        file_name = curr_dir + str(i) + ".txt"
        with open(file_name, 'w+') as fd:
            fd.write("0 %f %f %f %f\n" %
                     (xp  / sw, yp / sh, w / sw, h / sh))

    print("Dataset folder: " + outdir)

def run_labeling(outdir):
    """Begin data labeling sequence"""

    with open(os.path.join(outdir, "metadata.json"), "r") as metafile:
        metadata = json.load(metafile)

    set_names = [float(x) for x in list(metadata.keys()) if x != 'total_images']
    to_label = str(max(set_names))

    gen_labels(metadata[to_label], metadata['total_images'])
