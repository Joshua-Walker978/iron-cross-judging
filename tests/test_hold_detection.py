"""Unit and verification tests for src.hold_detection.

The verification tests build synthetic arm-angle signals whose hold start,
end and angle are known exactly (a "manufactured solution") and check that
the pipeline recovers them.
"""

import numpy as np
import pandas as pd
import pytest

from src.hold_detection import (
    add_angular_velocity,
    build_candidate_holds,
    choose_window_length,
    classify_hold_duration,
    detect_hold,
    find_stopped_segments,
    flag_stopped_frames,
    judge_hold,
    merge_segments,
    select_main_hold,
    smooth_deviations,
    timing_errors,
)

FPS = 30.0


def make_series(angle_fn, duration_s, fps=FPS, noise_deg=0.0, nan_frames=(), seed=0):
    """Per-frame deviation table from an angle profile angle_fn(t) in degrees."""
    n = int(round(duration_s * fps))
    t = np.arange(n) / fps
    rng = np.random.default_rng(seed)
    left = np.array([angle_fn(ti) for ti in t]) + rng.normal(0, noise_deg, n)
    right = np.array([angle_fn(ti) for ti in t]) + 1.0 + rng.normal(0, noise_deg, n)
    left[list(nan_frames)] = np.nan
    right[list(nan_frames)] = np.nan
    return pd.DataFrame(
        {
            "frame": np.arange(n),
            "time_sec": t,
            "left_deviation": left,
            "right_deviation": right,
            "detected": ~np.isnan(left),
        }
    )


def cross_profile(t, hold_start=2.0, hold_end=4.5, hold_angle=4.0, hang_angle=85.0):
    """Hang, lower into the cross over 1 s, hold, rise over 1 s, hang."""
    if t < hold_start - 1.0:
        return hang_angle
    if t < hold_start:
        return hang_angle + (hold_angle - hang_angle) * (t - (hold_start - 1.0))
    if t <= hold_end:
        return hold_angle
    if t < hold_end + 1.0:
        return hold_angle + (hang_angle - hold_angle) * (t - hold_end)
    return hang_angle


# --- small building blocks -------------------------------------------------


@pytest.mark.parametrize("n, expected", [(141, 9), (9, 9), (8, 7), (7, 7), (6, 5), (5, 5)])
def test_choose_window_length(n, expected):
    assert choose_window_length(n) == expected


def test_choose_window_length_rejects_tiny_clips():
    with pytest.raises(ValueError):
        choose_window_length(4)


def test_smoothing_leaves_a_straight_line_unchanged():
    """A Savitzky-Golay filter of order >= 1 reproduces a linear signal exactly."""
    series = make_series(lambda t: 10 + 5 * t, 2.0)
    out = smooth_deviations(series)
    np.testing.assert_allclose(out["left_smooth"], series["left_deviation"], atol=1e-9)


def test_smoothing_interpolates_interior_gaps():
    series = make_series(lambda t: 10 + 5 * t, 2.0, nan_frames=[20, 21, 22])
    out = smooth_deviations(series)
    assert not out["left_smooth"].isna().any()
    expected = 10 + 5 * series["time_sec"]
    np.testing.assert_allclose(out["left_smooth"], expected, atol=1e-9)


def test_smoothing_handles_missed_first_and_last_frames():
    """Regression test: a missed detection in frame 0 used to crash the Savitzky-Golay filter."""
    series = make_series(lambda t: 30.0, 2.0, nan_frames=[0, 1, 59])
    out = smooth_deviations(series)
    np.testing.assert_allclose(out["left_smooth"], 30.0, atol=1e-9)


def test_smoothing_rejects_clip_with_no_detections():
    series = make_series(lambda t: 30.0, 1.0, nan_frames=range(30))
    with pytest.raises(ValueError):
        smooth_deviations(series)


def test_velocity_of_a_linear_signal_is_its_slope():
    series = make_series(lambda t: 10 + 30 * t, 2.0)
    out = add_angular_velocity(smooth_deviations(series))
    np.testing.assert_allclose(out["left_velocity"], 30.0, atol=1e-6)


def test_stopped_needs_both_arms_below_threshold():
    df = pd.DataFrame({"left_velocity": [0.0, 20.0, 0.0, np.nan], "right_velocity": [0.0, 0.0, -20.0, 0.0]})
    assert flag_stopped_frames(df, 15.0)["is_stopped"].tolist() == [True, False, False, False]


def test_find_stopped_segments_groups_consecutive_frames():
    df = pd.DataFrame(
        {
            "time_sec": np.arange(8) / 10,
            "is_stopped": [False, True, True, False, True, True, True, False],
            "left_smooth": 1.0,
            "right_smooth": 3.0,
        }
    )
    segs = find_stopped_segments(df)
    assert segs["start_time"].tolist() == pytest.approx([0.1, 0.4])
    assert segs["end_time"].tolist() == pytest.approx([0.2, 0.6])
    assert segs["n_frames"].tolist() == [2, 3]
    assert segs["duration"].tolist() == pytest.approx([0.1, 0.2])


def test_find_stopped_segments_handles_no_stops():
    df = pd.DataFrame({"time_sec": [0.0, 0.1], "is_stopped": [False, False], "left_smooth": 0.0, "right_smooth": 0.0})
    assert find_stopped_segments(df).empty


def test_merge_segments_bridges_only_short_gaps():
    segs = pd.DataFrame({"start_time": [0.0, 1.2, 3.0], "end_time": [1.0, 2.0, 3.5]})
    assert merge_segments(segs, gap_tolerance=0.25) == [(0.0, 2.0), (3.0, 3.5)]
    assert merge_segments(segs, gap_tolerance=0.1) == [(0.0, 1.0), (1.2, 2.0), (3.0, 3.5)]


def test_merge_segments_empty():
    assert merge_segments(pd.DataFrame({"start_time": [], "end_time": []})) == []


def test_select_main_hold_prefers_longest_plausible_candidate():
    candidates = pd.DataFrame(
        {
            "start": [0.0, 1.0, 5.0],
            "end": [3.0, 2.0, 5.5],
            "duration": [3.0, 1.0, 0.5],
            "avg_angle": [80.0, 10.0, 12.0],  # the longest one is a hang, not a cross
        }
    )
    hold = select_main_hold(candidates)
    assert hold["start"] == 1.0


def test_select_main_hold_returns_none_when_nothing_qualifies():
    assert select_main_hold(build_candidate_holds(pd.DataFrame(), [])) is None
    candidates = pd.DataFrame({"start": [0.0], "end": [3.0], "duration": [3.0], "avg_angle": [60.0]})
    assert select_main_hold(candidates) is None


@pytest.mark.parametrize(
    "duration, expected",
    [
        (2.0, "No deduction"),
        (4.8, "No deduction"),
        (1.99, "Medium error"),
        (0.03, "Medium error"),
        (0.0, "Large error and non-recognition (no stop)"),
    ],
)
def test_classify_hold_duration(duration, expected):
    assert classify_hold_duration(duration) == expected


@pytest.mark.parametrize("bad", [float("nan"), -0.5])
def test_classify_hold_duration_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        classify_hold_duration(bad)


def test_timing_errors_arithmetic():
    hold = pd.Series({"start": 1.0, "end": 3.0, "duration": 2.0})
    err = timing_errors(hold, manual_start_frame=30, manual_end_frame=96, fps=30.0)
    assert err["manual_duration"] == pytest.approx(2.2)
    assert err["start_error"] == pytest.approx(0.0)
    assert err["end_error"] == pytest.approx(-0.2)
    assert err["duration_error"] == pytest.approx(-0.2)


def test_judge_hold_none():
    assert all(value is None for value in judge_hold(None).values())


def test_detect_hold_reports_missing_columns():
    with pytest.raises(KeyError):
        detect_hold(pd.DataFrame({"time_sec": [0.0] * 10}))


# --- verification against manufactured solutions ----------------------------


@pytest.mark.parametrize("hold_angle", [0.0, 4.0, 15.0, 30.0])
def test_recovers_known_hold(hold_angle):
    """A clean synthetic cross held from 2.0 s to 4.5 s at a known angle."""
    series = make_series(lambda t: cross_profile(t, hold_angle=hold_angle), 6.5)
    _, hold = detect_hold(series)
    assert hold is not None
    assert hold["start"] == pytest.approx(2.0, abs=0.2)
    assert hold["end"] == pytest.approx(4.5, abs=0.2)
    assert hold["duration"] == pytest.approx(2.5, abs=0.3)
    # right arm is offset by +1 deg, so the two-arm mean is hold_angle + 0.5
    assert hold["avg_angle"] == pytest.approx(hold_angle + 0.5, abs=1.0)


def test_recovers_known_hold_with_noise_and_dropouts():
    """Same hold with 0.5 deg landmark jitter and missed detections mid-hold."""
    series = make_series(cross_profile, 6.5, noise_deg=0.5, nan_frames=[90, 91, 92, 110])
    _, hold = detect_hold(series)
    assert hold["start"] == pytest.approx(2.0, abs=0.25)
    assert hold["end"] == pytest.approx(4.5, abs=0.25)
    assert hold["avg_angle"] == pytest.approx(4.5, abs=1.0)
    verdict = judge_hold(hold)
    assert verdict["angle_deduction"] == "No deduction"
    assert verdict["duration_deduction"] == "No deduction"


def test_short_hold_is_classified_as_medium_duration_error():
    series = make_series(lambda t: cross_profile(t, hold_end=3.2), 5.5)
    _, hold = detect_hold(series)
    assert hold["duration"] == pytest.approx(1.2, abs=0.3)
    assert judge_hold(hold)["duration_deduction"] == "Medium error"


def test_static_hang_is_not_a_cross():
    """Arms held still but near vertical must not be reported as a hold."""
    _, hold = detect_hold(make_series(lambda t: 85.0, 3.0))
    assert hold is None


def test_continuous_motion_gives_no_hold():
    """Arms sweeping steadily at 45 deg/s never stop."""
    _, hold = detect_hold(make_series(lambda t: 90 - 45 * t, 2.0))
    assert hold is None
