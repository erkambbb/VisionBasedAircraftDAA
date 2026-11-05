"""Tag finder: works on raw 1920x1080 frames and saved 1280x720 frames, ignores flat sky."""
import cv2
import numpy as np

from data_generation import overlay_clean as oc


def _bg(w, h):
    return np.random.default_rng(0).integers(120, 135, (h, w, 3), dtype=np.uint8)


def _tag():
    return cv2.imread(oc.REF_PATH)


def test_finds_tag_on_saved_size_frame():
    f = _bg(1280, 720)
    f[200:226, 400:502] = _tag()
    x, y, w, h, score = oc.find_label(f)
    assert score > 0.8 and abs(x - 406) <= 2 and abs(y - 207) <= 2 and (w, h) == (88, 11)


def test_finds_tag_on_raw_size_frame():
    f = _bg(1920, 1080)                                  # X-Plane draws it 1.5x larger than in saved frames
    f[300:339, 600:753] = cv2.resize(_tag(), (153, 39), interpolation=cv2.INTER_CUBIC)
    x, y, w, h, score = oc.find_label(f)
    assert score > 0.8 and abs(x - 609) <= 3 and abs(y - 310) <= 3 and abs(w - 132) <= 1 and abs(h - 16) <= 1


def test_remove_label_on_raw_frame_restores_empty_frame():
    b = _bg(1920, 1080)
    a = b.copy()
    a[300:339, 600:753] = cv2.resize(_tag(), (153, 39), interpolation=cv2.INTER_CUBIC)
    out, status = oc.remove_label(a, b)
    assert status == "removed"
    assert np.array_equal(out[311:327, 609:741], b[311:327, 609:741])   # the tag itself, without the crop margin


def test_flat_sky_is_not_a_tag():
    assert oc.find_label(np.full((720, 1280, 3), 205, np.uint8)) is None
    f = np.full((1080, 1920, 3), 205, np.uint8)
    f[800:, :] = 90                                      # horizon, rest of the frame flat
    assert not oc.label_left(f)
