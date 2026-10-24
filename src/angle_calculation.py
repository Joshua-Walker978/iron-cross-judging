"""
Arm angle measurement and angle-deduction classification.

Functions here take MediaPipe pose landmarks, convert them to pixel
coordinates, measure how far each arm deviates from a horizontal line, and
classify that deviation into the angle-deviation bands that the FIG Men's
Artistic Gymnastics Code of Points (2025-2028) applies to strength hold
elements on still rings.

Coordinate convention
---------------------
Pixel coordinates follow the image convention: the origin is the top-left
corner, x increases to the right and y increases downwards. All angles are
in degrees.
"""

__docformat__ = "numpy"

import numpy as np

# MediaPipe Pose landmark indices (33-point BlazePose topology).
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

# Upper limit (inclusive, degrees) of each angle-deviation band.
NO_DEDUCTION_MAX_DEG = 5.0
SMALL_ERROR_MAX_DEG = 20.0
MEDIUM_ERROR_MAX_DEG = 45.0

DEDUCTION_POINTS = {
    "No deduction": 0.0,
    "Small error": 0.1,
    "Medium error": 0.3,
    "Large error": 0.5,
}
"""Execution deduction in points for each category returned by
`classify_deduction`. A large angle error (> 45 deg) also means the element
is not recognised by the D-jury."""


def get_point(landmarks, index, width, height):
    """
    Convert one normalised MediaPipe landmark into pixel coordinates.

    Parameters
    ----------
    landmarks : sequence
        Landmarks for one person, i.e. ``result.pose_landmarks[0]`` from a
        MediaPipe ``PoseLandmarkerResult``. Each element must have ``x`` and
        ``y`` attributes normalised to [0, 1] by image width and height.
    index : int
        Index of the landmark to extract, e.g. `LEFT_SHOULDER`.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.

    Returns
    -------
    numpy.ndarray
        Pixel coordinates ``[x, y]``, shape ``(2,)``.

    Raises
    ------
    IndexError
        If `index` is outside the landmark list.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> landmarks = [SimpleNamespace(x=0.5, y=0.25)]
    >>> get_point(landmarks, 0, width=200, height=400)
    array([100., 100.])
    """
    lm = landmarks[index]
    return np.array([lm.x * width, lm.y * height])


def get_arm_points(landmarks, width, height):
    """
    Extract both shoulders and both wrists in pixel coordinates.

    Parameters
    ----------
    landmarks : sequence
        Landmarks for one person (see `get_point`).
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.

    Returns
    -------
    dict of str to numpy.ndarray
        Keys ``"left_shoulder"``, ``"right_shoulder"``, ``"left_wrist"`` and
        ``"right_wrist"``; each value is an ``[x, y]`` pixel array.
    """
    return {
        "left_shoulder": get_point(landmarks, LEFT_SHOULDER, width, height),
        "right_shoulder": get_point(landmarks, RIGHT_SHOULDER, width, height),
        "left_wrist": get_point(landmarks, LEFT_WRIST, width, height),
        "right_wrist": get_point(landmarks, RIGHT_WRIST, width, height),
    }


def deviation_from_horizontal(shoulder, wrist):
    """
    Angle between the shoulder-to-wrist line and the horizontal.

    The signed angle of the vector from shoulder to wrist is folded into
    [0, 90] degrees, so the result is the same whether the arm points left
    or right in the image and whether it sits above or below the shoulder
    line.

    Parameters
    ----------
    shoulder : array_like
        ``[x, y]`` pixel coordinates of the shoulder.
    wrist : array_like
        ``[x, y]`` pixel coordinates of the wrist.

    Returns
    -------
    float
        Deviation from horizontal in degrees, in the range [0, 90].

    Notes
    -----
    The angle is measured in the 2D image plane. Any rotation of the athlete
    away from the camera (yaw) foreshortens the horizontal component of the
    arm and makes the deviation read larger than it really is; see
    `src.perspective`. Because the angle is folded, arms raised above and
    arms dropped below horizontal by the same amount give the same value.
    If shoulder and wrist coincide the result is 0, which is meaningless;
    this only happens with a failed detection.

    Examples
    --------
    >>> deviation_from_horizontal([0, 0], [10, 0])
    0.0
    >>> deviation_from_horizontal([0, 0], [-10, 10])
    45.0
    >>> deviation_from_horizontal([0, 0], [0, 10])
    90.0
    """
    dx = wrist[0] - shoulder[0]
    dy = wrist[1] - shoulder[1]
    angle_deg = np.degrees(np.arctan2(dy, dx))

    deviation = abs(angle_deg)
    if deviation > 90:
        deviation = 180 - deviation
    return float(deviation)


def classify_deduction(deviation_degrees):
    """
    Classify an arm's deviation from horizontal into a deduction category.

    Bands follow the 2025-2028 FIG MAG Code of Points for strength holds:

    ===============  =================  =========
    Deviation        Category           Deduction
    ===============  =================  =========
    0 to 5 deg       No deduction       0.0
    > 5 to 20 deg    Small error        0.1
    > 20 to 45 deg   Medium error       0.3
    > 45 deg         Large error        0.5 and element not recognised
    ===============  =================  =========

    Parameters
    ----------
    deviation_degrees : float
        Deviation from horizontal in degrees, normally from
        `deviation_from_horizontal`.

    Returns
    -------
    str
        One of ``"No deduction"``, ``"Small error"``, ``"Medium error"`` or
        ``"Large error"``. Use `DEDUCTION_POINTS` to convert to points.

    Raises
    ------
    ValueError
        If `deviation_degrees` is NaN or negative. NaN usually means no pose
        was detected, which should not be silently scored as a large error.

    Examples
    --------
    >>> classify_deduction(5.0)
    'No deduction'
    >>> classify_deduction(28.1)
    'Medium error'
    """
    if deviation_degrees is None or np.isnan(deviation_degrees):
        raise ValueError("deviation_degrees is NaN or None (no pose detected?)")
    if deviation_degrees < 0:
        raise ValueError(f"deviation_degrees must be >= 0, got {deviation_degrees}")

    if deviation_degrees <= NO_DEDUCTION_MAX_DEG:
        return "No deduction"
    elif deviation_degrees <= SMALL_ERROR_MAX_DEG:
        return "Small error"
    elif deviation_degrees <= MEDIUM_ERROR_MAX_DEG:
        return "Medium error"
    else:
        return "Large error"
