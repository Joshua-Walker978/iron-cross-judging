"""
Angle calculation and deduction classification for iron cross judging.

Functions here take pose landmark coordinates (from MediaPipe PoseLandmarker)
and compute arm deviation from horizontal, then classify the result according
to the FIG Code of Points deduction bands for strength hold positions.
"""

import numpy as np

# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


def get_point(landmarks, index, width, height):
    """
    Convert a normalized MediaPipe landmark into pixel coordinates.

    Args:
        landmarks: list of MediaPipe pose landmarks (normalized 0-1 coords).
        index (int): landmark index to extract (see module constants).
        width (int): image width in pixels.
        height (int): image height in pixels.

    Returns:
        np.ndarray: [x, y] pixel coordinates of the landmark.
    """
    lm = landmarks[index]
    return np.array([lm.x * width, lm.y * height])


def deviation_from_horizontal(shoulder, wrist):
    """
    Compute the angular deviation of an arm from a perfectly horizontal line.

    Args:
        shoulder (np.ndarray): [x, y] pixel coordinates of the shoulder.
        wrist (np.ndarray): [x, y] pixel coordinates of the wrist.

    Returns:
        float: deviation from horizontal, in degrees, always in [0, 90].
    """
    dx = wrist[0] - shoulder[0]
    dy = wrist[1] - shoulder[1]
    angle_deg = np.degrees(np.arctan2(dy, dx))

    deviation = abs(angle_deg)
    if deviation > 90:
        deviation = 180 - deviation
    return deviation


def classify_deduction(deviation_degrees):
    """
    Classify an arm's angular deviation into a FIG deduction band.

    Bands (per FIG Code of Points, strength hold positions):
        <=5deg   : No deduction
        <=20deg  : Small error
        <=45deg  : Medium error
        >45deg   : Large error

    Args:
        deviation_degrees (float): deviation from horizontal, in degrees.

    Returns:
        str: one of "No deduction", "Small error", "Medium error", "Large error".
    """
    if deviation_degrees <= 5:
        return "No deduction"
    elif deviation_degrees <= 20:
        return "Small error"
    elif deviation_degrees <= 45:
        return "Medium error"
    else:
        return "Large error"
