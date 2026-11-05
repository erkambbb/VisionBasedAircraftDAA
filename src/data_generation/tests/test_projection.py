import math
import numpy as np
import pytest
from data_generation.projection import (fov_from_projection, box_corners_ogl,
                                        project_points, bbox_from_points)
from data_generation.constants import AIRCRAFT_BOX

SW, SH = 1280, 720
F_PX = 640 / math.tan(math.radians(29.36 / 2))
I16 = list(np.eye(4).flatten())
BOX = AIRCRAFT_BOX["Shadow Twin Boom"]


def make_proj(hfov_deg, aspect, near=1.0, far=1e5):
    tx = math.tan(math.radians(hfov_deg) / 2)
    f = 1.0 / (tx / aspect)          # 1/tan(vfov/2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return list(m.T.flatten())       # column-major


def test_fov_from_projection_roundtrip():
    h, v = fov_from_projection(make_proj(29.36, 16 / 9))
    assert abs(h - 29.36) < 0.01 and abs(v - 16.76) < 0.02


def test_corners_shape_and_center():
    c = box_corners_ogl((0, 0, 0), 0, 0, 0, (-1, -1, -1), (1, 1, 1))
    assert c.shape == (8, 3) and np.allclose(c.mean(axis=0), 0)


def test_orientation_signs():
    nose = box_corners_ogl((0, 0, 0), 90, 0, 0, (0, 0, -1), (0, 0, -1))[0]
    assert np.allclose(nose, (1, 0, 0), atol=1e-6)
    wing = box_corners_ogl((0, 0, 0), 90, 0, 0, (1, 0, 0), (1, 0, 0))[0]
    assert np.allclose(wing, (0, 0, 1), atol=1e-6)
    up = box_corners_ogl((0, 0, 0), 0, 10, 0, (0, 0, -1), (0, 0, -1))[0]
    assert up[1] > 0
    down = box_corners_ogl((0, 0, 0), 0, 0, 10, (1, 0, 0), (1, 0, 0))[0]
    assert down[1] < 0


def _bbox(center, psi=0.0):
    corners = box_corners_ogl(center, psi, 0, 0, *BOX)
    px, front = project_points(corners, I16, make_proj(29.36, 16 / 9), SW, SH)
    return bbox_from_points(px, front, SW, SH)


def test_rear_view_width_matches_span():
    cx, cy, w, h = _bbox((0, 0, -100))
    exp = F_PX * 3.8 / 100
    assert abs(w - exp) / exp < 0.03
    assert abs(cx - 640) < 1.5


def test_side_view_width_matches_length():
    cx, cy, w, h = _bbox((0, 0, -100), psi=90)
    exp = F_PX * 2.408 / 100
    assert abs(w - exp) / exp < 0.05


def test_behind_camera_returns_none():
    assert _bbox((0, 0, 100)) is None


def test_outside_frame_returns_none():
    assert _bbox((200, 0, -100)) is None


def test_half_clipped_box_rules():
    # 100 m: x px = 640 + F_PX*x/100 ; right edge at x=640px -> x_m = 640*100/F_PX
    edge = 640 * 100 / F_PX
    w_m = 3.8
    # box mostly outside (visible ~0.3)
    assert _bbox((edge + w_m / 2 - 0.3 * w_m, 0, -100)) is None
    r = _bbox((edge + w_m / 2 - 0.7 * w_m, 0, -100))
    assert r is not None and r[0] + r[2] / 2 <= SW + 1e-6
