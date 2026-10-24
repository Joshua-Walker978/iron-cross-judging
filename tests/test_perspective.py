"""Tests for src.perspective using a synthetic body rotated by a known yaw."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.angle_calculation import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    deviation_from_horizontal,
)
from src.perspective import (
    corrected_deviation,
    estimate_cos_yaw,
    estimate_yaw_and_correct,
    get_calibration_ratio,
)

SIZE = 1000  # square test image, pixels
CENTRE_X = 0.5


def body(yaw_deg):
    """Normalised landmarks of a front-on torso, foreshortened horizontally by cos(yaw)."""
    c = np.cos(np.radians(yaw_deg))
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(33)]
    for index, (x, y) in {
        LEFT_SHOULDER: (0.6, 0.3),
        RIGHT_SHOULDER: (0.4, 0.3),
        LEFT_HIP: (0.58, 0.6),
        RIGHT_HIP: (0.42, 0.6),
    }.items():
        landmarks[index] = SimpleNamespace(x=CENTRE_X + (x - CENTRE_X) * c, y=y)
    return landmarks


def projected_arm(true_deviation_deg, yaw_deg, length=300.0):
    """Shoulder and wrist pixels of an arm at a true deviation, seen with a given yaw."""
    c = np.cos(np.radians(yaw_deg))
    rad = np.radians(true_deviation_deg)
    shoulder = np.array([500.0, 300.0])
    wrist = shoulder + np.array([length * np.cos(rad) * c, length * np.sin(rad)])
    return shoulder, wrist


def test_calibration_ratio_of_reference_body():
    # shoulder width 200 px, torso length 300 px
    assert get_calibration_ratio(body(0), SIZE, SIZE) == pytest.approx(200 / 300)


def test_calibration_ratio_rejects_zero_torso():
    flat = [SimpleNamespace(x=0.5, y=0.5) for _ in range(33)]
    with pytest.raises(ValueError):
        get_calibration_ratio(flat, SIZE, SIZE)


@pytest.mark.parametrize("yaw", [0, 20, 45, 60])
def test_yaw_is_recovered_when_proportions_match(yaw):
    reference = get_calibration_ratio(body(0), SIZE, SIZE)
    current = get_calibration_ratio(body(yaw), SIZE, SIZE)
    assert np.degrees(np.arccos(estimate_cos_yaw(current, reference))) == pytest.approx(yaw, abs=1e-6)


def test_cos_yaw_is_clipped():
    assert estimate_cos_yaw(1.5, 1.0) == 1.0
    assert estimate_cos_yaw(0.01, 1.0) == pytest.approx(0.1)


def test_correction_with_zero_yaw_changes_nothing():
    shoulder, wrist = projected_arm(12.0, 0)
    assert corrected_deviation(shoulder, wrist, 1.0) == pytest.approx(deviation_from_horizontal(shoulder, wrist))


@pytest.mark.parametrize("true_deviation, yaw", [(0, 40), (5, 30), (10, 60), (20, 45)])
def test_full_correction_recovers_true_angle(true_deviation, yaw):
    """With matching body proportions the corrected angle equals the true angle."""
    reference = get_calibration_ratio(body(0), SIZE, SIZE)
    shoulder, wrist = projected_arm(true_deviation, yaw)
    raw = deviation_from_horizontal(shoulder, wrist)
    corrected, yaw_est = estimate_yaw_and_correct(shoulder, wrist, reference, body(yaw), SIZE, SIZE)
    assert yaw_est == pytest.approx(yaw, abs=1e-6)
    assert corrected == pytest.approx(true_deviation, abs=1e-6)
    if true_deviation > 0:
        assert raw > corrected  # yaw makes the arm look steeper
