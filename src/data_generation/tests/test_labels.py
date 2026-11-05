import os
import pandas as pd
from data_generation.label_traffic_data import yolo_line, gen_labels


def test_yolo_line_normalizes_by_raw_size():
    assert yolo_line(960, 540, 96, 54, 1920, 1080) == "0 0.500000 0.500000 0.050000 0.050000\n"


def _setup(tmp_path, extra):
    for d in ("train", "valid"):
        os.makedirs(tmp_path / "labels" / d)
        os.makedirs(tmp_path / "images" / d)
    row = dict(e0=0, n0=0, u0=0, h0=0, e1=0, n1=100, u1=0, h1=0, intr_x=960, intr_y=540,
               raw_w=1920, raw_h=1080)
    row.update(extra)
    df = pd.DataFrame([row, row])
    df.to_csv(tmp_path / "state_data.csv", index=False)
    return {'outdir': str(tmp_path), 'num_train': 1, 'num_valid': 1, 'ac': 'Cessna Skyhawk'}


def test_gen_labels_uses_bb_columns(tmp_path):
    md = _setup(tmp_path, dict(bb_cx=960, bb_cy=540, bb_w=96, bb_h=54))
    gen_labels(md, 2)
    exp = "0 0.500000 0.500000 0.050000 0.050000\n"
    assert (tmp_path / "labels/train/0.txt").read_text() == exp
    assert (tmp_path / "labels/valid/1.txt").read_text() == exp


def test_legacy_dataset_still_labels(tmp_path):
    md = _setup(tmp_path, {})
    gen_labels(md, 2)
    assert (tmp_path / "labels/train/0.txt").exists()
