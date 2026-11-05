"""Camera geometry helpers.

Used to make the simulated dataset match a real camera's field of view.
Default sensor: Sony IMX477R (Raspberry Pi High Quality Camera).
  full active array 4056 x 3040 px, pixel size 1.55 um
"""
import math

IMX477_PIXEL_UM = 1.55
IMX477_FULL_W_PX = 4056
IMX477_FULL_H_PX = 3040


def sensor_size_mm(crop_16_9=True):
    """Returns (width_mm, height_mm) of the sensor area used for the image.

    crop_16_9=True keeps the full width and cuts top/bottom to 16:9,
    which matches the sample frames of the real camera.
    """
    w_mm = IMX477_FULL_W_PX * IMX477_PIXEL_UM / 1000.0
    if crop_16_9:
        h_mm = w_mm * 9.0 / 16.0
    else:
        h_mm = IMX477_FULL_H_PX * IMX477_PIXEL_UM / 1000.0
    return w_mm, h_mm


def camera_fov(focal_mm, crop_16_9=True):
    """Pinhole field of view in degrees for the given lens focal length in mm.

    Returns
    -------
    hfov, vfov : float
        horizontal and vertical field of view in degrees
    """
    if focal_mm <= 0:
        raise ValueError("focal_mm must be positive")
    w_mm, h_mm = sensor_size_mm(crop_16_9)
    hfov = 2 * math.degrees(math.atan(w_mm / (2 * focal_mm)))
    vfov = 2 * math.degrees(math.atan(h_mm / (2 * focal_mm)))
    return hfov, vfov
