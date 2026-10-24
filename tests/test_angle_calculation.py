"""Unit tests for src.angle_calculation."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.angle_calculation import (
    DEDUCTION_POINTS,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    classify_deduction,
    deviation_from_horizontal,
    get_arm_points,
    get_point,
)


def make_landmarks(points):
    """Build a 33-landmark list with the given {index: (x, y)} normalised positions."""
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(33)]
    for index, (x, y) in points.items():
        landmarks[index] = SimpleNamespace(x=x, y=y)
    return landmarks


# --- get_point / get_arm_points -------------------------------------------


def test_get_point_scales_by_width_and_height_separately():
    landmarks = make_landmarks({LEFT_SHOULDER: (0.25, 0.5)})
    np.testing.assert_allclose(get_point(landmarks, LEFT_SHOULDER, 400, 1000), [100.0, 500.0])


def test_get_point_rejects_bad_index():
    with pytest.raises(IndexError):
        get_point(make_landmarks({}), 40, 100, 100)


def test_get_arm_points_returns_all_four_points():
    landmarks = make_landmarks(
        {
            LEFT_SHOULDER: (0.6, 0.3),
            RIGHT_SHOULDER: (0.4, 0.3),
            LEFT_WRIST: (0.9, 0.3),
            RIGHT_WRIST: (0.1, 0.3),
        }
    )
    pts = get_arm_points(landmarks, 100, 100)
    assert set(pts) == {"left_shoulder", "right_shoulder", "left_wrist", "right_wrist"}
    np.testing.assert_allclose(pts["right_wrist"], [10.0, 30.0])


# --- deviation_from_horizontal -----------------------------------------------


@pytest.mark.parametrize(
    "wrist, expected",
    [
        ((10, 0), 0.0),  # horizontal, pointing right
        ((-10, 0), 0.0),  # horizontal, pointing left
        ((0, 10), 90.0),  # straight down (y grows downwards)
        ((0, -10), 90.0),  # straight up
        ((10, 10), 45.0),
        ((-10, -10), 45.0),
    ],
)
def test_deviation_known_angles(wrist, expected):
    assert deviation_from_horizontal((0, 0), wrist) == pytest.approx(expected)


@pytest.mark.parametrize("angle", [0.0, 3.0, 14.8, 30.0, 60.0, 89.0])
def test_deviation_is_symmetric_for_left_and_right_arms(angle):
    """Mirror-image arms at the same angle must give the same deviation."""
    rad = np.radians(angle)
    left = deviation_from_horizontal((0, 0), (np.cos(rad), np.sin(rad)))
    right = deviation_from_horizontal((0, 0), (-np.cos(rad), np.sin(rad)))
    assert left == pytest.approx(angle)
    assert right == pytest.approx(angle)


def test_deviation_is_independent_of_arm_length_and_position():
    a = deviation_from_horizontal((100, 200), (110, 205))
    b = deviation_from_horizontal((0, 0), (1000, 500))
    assert a == pytest.approx(b)


def test_deviation_range_on_random_vectors():
    rng = np.random.default_rng(0)
    for dx, dy in rng.normal(size=(500, 2)):
        assert 0.0 <= deviation_from_horizontal((0, 0), (dx, dy)) <= 90.0


# --- classify_deduction ------------------------------------------------------


@pytest.mark.parametrize(
    "deviation, expected",
    [
        (0.0, "No deduction"),
        (5.0, "No deduction"),  # band limits are inclusive
        (5.01, "Small error"),
        (20.0, "Small error"),
        (20.01, "Medium error"),
        (45.0, "Medium error"),
        (45.01, "Large error"),
        (90.0, "Large error"),
    ],
)
def test_classify_deduction_band_edges(deviation, expected):
    assert classify_deduction(deviation) == expected


@pytest.mark.parametrize("bad", [float("nan"), None, -1.0])
def test_classify_deduction_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        classify_deduction(bad)


def test_every_category_has_a_point_value():
    for deviation in (0, 10, 30, 60):
        assert classify_deduction(deviation) in DEDUCTION_POINTS
