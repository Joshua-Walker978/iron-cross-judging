"""
Experimental camera-yaw correction based on body proportions.

When the camera is not square to the athlete, the athlete is rotated about
the vertical axis (yaw) relative to the image plane. Horizontal lengths,
such as shoulder width and the horizontal reach of the arms, shrink by
roughly ``cos(yaw)``, while vertical lengths such as the torso do not. The
arm therefore looks steeper than it is and the measured deviation from
horizontal is too large.

This module estimates ``cos(yaw)`` by comparing the athlete's
shoulder-width to torso-length ratio with the same ratio measured in a
front-on reference image, then stretches the horizontal component of the
arm vector back out before measuring the angle.

Assumptions
-----------
- Orthographic projection (camera far from the athlete compared with the
  athlete's size).
- Yaw is the only rotation; pitch and roll of the camera are ignored.
- The athlete and the reference person have the same body proportions.
  This is the weakest assumption: a broader-shouldered athlete reads as
  "less rotated" than they are, and if their ratio exceeds the reference
  the estimate is clipped to zero yaw and no correction is applied.
- The correction cannot tell whether the athlete is turned left or right.
"""

__docformat__ = "numpy"

import numpy as np

from src.angle_calculation import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    get_point,
)

MIN_COS_YAW = 0.1
"""Lower clip on cos(yaw) (about 84 deg) so the correction cannot divide by ~0."""


def get_calibration_ratio(landmarks, width, height):
    """
    Shoulder width divided by torso length, in the image plane.

    Parameters
    ----------
    landmarks : sequence
        Landmarks for one person (see `src.angle_calculation.get_point`).
    width : int
        Width in pixels of the image the landmarks came from.
    height : int
        Height in pixels of the image the landmarks came from.

    Returns
    -------
    float
        Distance between the shoulders divided by the distance from the
        shoulder midpoint to the hip midpoint (dimensionless).

    Raises
    ------
    ValueError
        If the torso length is zero (failed detection).

    Notes
    -----
    `width` and `height` must belong to the same image as `landmarks`.
    MediaPipe normalises x by width and y by height separately, so pairing
    landmarks from one image with the size of another changes the ratio.
    """
    ls = get_point(landmarks, LEFT_SHOULDER, width, height)
    rs = get_point(landmarks, RIGHT_SHOULDER, width, height)
    lh = get_point(landmarks, LEFT_HIP, width, height)
    rh = get_point(landmarks, RIGHT_HIP, width, height)

    shoulder_width = np.linalg.norm(ls - rs)
    torso_length = np.linalg.norm((ls + rs) / 2 - (lh + rh) / 2)
    if torso_length == 0:
        raise ValueError("Torso length is zero; check the pose detection")
    return float(shoulder_width / torso_length)


def estimate_cos_yaw(current_ratio, reference_ratio, min_cos=MIN_COS_YAW):
    """
    Estimate cos(yaw) from the current and reference body ratios.

    Parameters
    ----------
    current_ratio : float
        `get_calibration_ratio` for the frame being judged.
    reference_ratio : float
        `get_calibration_ratio` for a front-on (zero-yaw) image.
    min_cos : float, optional
        Lower clip on the result. Default 0.1.

    Returns
    -------
    float
        ``current_ratio / reference_ratio`` clipped to ``[min_cos, 1]``.

    Examples
    --------
    >>> round(estimate_cos_yaw(0.5, 1.0), 3)
    0.5
    >>> estimate_cos_yaw(1.2, 1.0)  # broader than the reference: clipped
    1.0
    """
    return float(np.clip(current_ratio / reference_ratio, min_cos, 1.0))


def corrected_deviation(shoulder, wrist, cos_yaw):
    """
    Deviation from horizontal after undoing horizontal foreshortening.

    Parameters
    ----------
    shoulder : array_like
        ``[x, y]`` pixel coordinates of the shoulder.
    wrist : array_like
        ``[x, y]`` pixel coordinates of the wrist.
    cos_yaw : float
        Estimated cos(yaw) from `estimate_cos_yaw`, in (0, 1].

    Returns
    -------
    float
        Corrected deviation from horizontal in degrees, in [0, 90]. With
        ``cos_yaw == 1`` this equals
        `src.angle_calculation.deviation_from_horizontal`.
    """
    dx = (wrist[0] - shoulder[0]) / cos_yaw
    dy = wrist[1] - shoulder[1]
    deviation = abs(np.degrees(np.arctan2(dy, dx)))
    if deviation > 90:
        deviation = 180 - deviation
    return float(deviation)


def estimate_yaw_and_correct(shoulder, wrist, reference_ratio, landmarks, width, height):
    """
    Estimate camera yaw for a frame and correct one arm's deviation.

    Parameters
    ----------
    shoulder : array_like
        ``[x, y]`` pixel coordinates of the arm's shoulder.
    wrist : array_like
        ``[x, y]`` pixel coordinates of the arm's wrist.
    reference_ratio : float
        `get_calibration_ratio` measured on a front-on reference image.
    landmarks : sequence
        Landmarks of the frame being corrected.
    width : int
        Width in pixels of that frame.
    height : int
        Height in pixels of that frame.

    Returns
    -------
    deviation : float
        Corrected deviation from horizontal in degrees.
    yaw_deg : float
        Estimated yaw in degrees, in [0, ~84]. Exactly 0 means the estimate
        was clipped and no correction was applied.
    """
    current_ratio = get_calibration_ratio(landmarks, width, height)
    cos_yaw = estimate_cos_yaw(current_ratio, reference_ratio)
    deviation = corrected_deviation(shoulder, wrist, cos_yaw)
    yaw_deg = float(np.degrees(np.arccos(cos_yaw)))
    return deviation, yaw_deg
