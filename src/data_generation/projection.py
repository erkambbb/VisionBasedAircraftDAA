"""Pure-numpy camera projection helpers (no X-Plane imports)."""
import itertools
import math
import numpy as np


def _mat(m):
    return np.reshape(np.asarray(m, dtype=float), (4, 4)).T


def fov_from_projection(proj):
    """(hfov_deg, vfov_deg) from a 16-element OpenGL column-major projection matrix."""
    p = _mat(proj)
    return (math.degrees(2 * math.atan(1.0 / p[0, 0])),
            math.degrees(2 * math.atan(1.0 / p[1, 1])))


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def body_points_ogl(points_body, center_ogl, psi_deg, the_deg, phi_deg):
    """Body-frame points (N,3) -> X-Plane local coords (x east, y up, z south).

    Body: x right wing, y up, z aft (nose = -z). Order: roll, pitch, heading.
    """
    pts = np.asarray(points_body, float).reshape(-1, 3)
    psi, the, phi = (math.radians(a) for a in (psi_deg, the_deg, phi_deg))
    # roll about nose axis (-z): +phi = right wing down; pitch about +x: +the = nose up;
    # heading: clockwise from above = negative rotation about +y
    r = _rot_y(-psi) @ _rot_x(the) @ _rot_z(-phi)
    return pts @ r.T + np.asarray(center_ogl, float)


def box_corners_ogl(center_ogl, psi_deg, the_deg, phi_deg, box_min, box_max):
    """8 corners (8,3) of a body-frame box in X-Plane local coords."""
    mn, mx = np.asarray(box_min, float), np.asarray(box_max, float)
    pts = np.array([[(mn, mx)[i][0], (mn, mx)[j][1], (mn, mx)[k][2]]
                    for i, j, k in itertools.product((0, 1), repeat=3)])
    return body_points_ogl(pts, center_ogl, psi_deg, the_deg, phi_deg)


def project_points(points_ogl, world_matrix, proj_matrix, sw, sh):
    """Returns (px (N,2) top-left origin, in_front (N,) bool)."""
    pts = np.asarray(points_ogl, float)
    h = np.hstack([pts, np.ones((len(pts), 1))])
    clip = (_mat(proj_matrix) @ (_mat(world_matrix) @ h.T)).T
    w = clip[:, 3]
    in_front = w > 0
    safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    ndc_x, ndc_y = clip[:, 0] / safe, clip[:, 1] / safe
    px = np.stack([sw * (ndc_x * 0.5 + 0.5), sh - sh * (ndc_y * 0.5 + 0.5)], axis=1)
    return px, in_front


def bbox_from_points(px, in_front, sw, sh, min_visible=0.5):
    """(cx, cy, w, h) in px, clipped to frame; None if behind camera or too little visible."""
    if not np.all(in_front):
        return None
    x0, y0 = px.min(axis=0)
    x1, y1 = px.max(axis=0)
    full = (x1 - x0) * (y1 - y0)
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, sw), min(y1, sh)
    if cx1 <= cx0 or cy1 <= cy0 or full <= 0:
        return None
    if (cx1 - cx0) * (cy1 - cy0) / full < min_visible:
        return None
    return ((cx0 + cx1) / 2, (cy0 + cy1) / 2, cx1 - cx0, cy1 - cy0)


def bbox_loose(px, in_front, sw, sh, min_visible=0.5, min_frame_cover=0.15):
    """Like bbox_from_points, but a large object that is cut by the frame is kept
    when the clipped box covers at least `min_frame_cover` of the frame (close-up UAV).
    Returns (cx, cy, w, h) or None."""
    if not np.all(in_front):
        return None
    x0, y0 = px.min(axis=0)
    x1, y1 = px.max(axis=0)
    full = (x1 - x0) * (y1 - y0)
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, sw), min(y1, sh)
    if cx1 <= cx0 or cy1 <= cy0 or full <= 0:
        return None
    clipped = (cx1 - cx0) * (cy1 - cy0)
    if clipped / full < min_visible and clipped / (sw * sh) < min_frame_cover:
        return None
    return ((cx0 + cx1) / 2, (cy0 + cy1) / 2, cx1 - cx0, cy1 - cy0)
