"""
Detect the held part of an iron cross from a per-frame arm-angle signal.

The input is a table with one row per video frame (normally produced by
`src.pose_estimation.extract_deviation_series`) holding the deviation from
horizontal of each arm. The processing chain is:

1. `smooth_deviations` - fill missed detections by linear interpolation,
   then apply a Savitzky-Golay filter to suppress landmark jitter.
2. `add_angular_velocity` - central-difference derivative of the smoothed
   angles with respect to time.
3. `flag_stopped_frames` - a frame is "stopped" when both arms move slower
   than a velocity threshold.
4. `find_stopped_segments` - group consecutive stopped frames.
5. `merge_segments` - join segments separated by short gaps (brief
   tremors or detection noise during a real hold).
6. `build_candidate_holds` and `select_main_hold` - keep segments whose mean
   angle could plausibly be a cross and take the longest one.

`detect_hold` runs the whole chain, and `judge_hold` converts the result to
deduction categories.

Default parameter values were chosen by inspecting the HPCComp.mov
recording at about 30 frames per second; see the module constants.
"""

__docformat__ = "numpy"

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.angle_calculation import classify_deduction

DEFAULT_WINDOW_LENGTH = 9
"""Savitzky-Golay window in frames (about 0.3 s at 30 fps)."""

DEFAULT_POLYORDER = 2
"""Savitzky-Golay polynomial order."""

MIN_WINDOW_LENGTH = 5
"""Smallest window used for short clips; fewer frames than this is an error."""

VELOCITY_THRESHOLD_DEG_S = 15.0
"""Both arms must move slower than this (deg/s) for a frame to count as stopped."""

GAP_TOLERANCE_S = 0.25
"""Stopped segments separated by at most this gap (s) are merged."""

MAX_CROSS_ANGLE_DEG = 45.0
"""Segments with a mean deviation above this (deg) are not treated as a cross."""

MIN_HOLD_DURATION_S = 2.0
"""Hold time required by the Code of Points for strength elements (s)."""

REQUIRED_COLUMNS = ("time_sec", "left_deviation", "right_deviation")


def choose_window_length(n_frames, preferred=DEFAULT_WINDOW_LENGTH, minimum=MIN_WINDOW_LENGTH):
    """
    Pick an odd Savitzky-Golay window that fits the clip length.

    Parameters
    ----------
    n_frames : int
        Number of frames in the clip.
    preferred : int, optional
        Window to use when the clip is long enough. Default 9.
    minimum : int, optional
        Smallest acceptable window. Default 5.

    Returns
    -------
    int
        ``min(preferred, largest odd number <= n_frames)``.

    Raises
    ------
    ValueError
        If `n_frames` is smaller than `minimum`.

    Examples
    --------
    >>> choose_window_length(141)
    9
    >>> choose_window_length(8)
    7
    """
    if n_frames < minimum:
        raise ValueError(f"Need at least {minimum} frames to smooth, got {n_frames}")
    largest_odd = n_frames if n_frames % 2 == 1 else n_frames - 1
    return max(min(preferred, largest_odd), minimum)


def _check_columns(df, columns):
    """Raise a clear error if `df` is missing any of `columns`."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing required column(s): {missing}")


def smooth_deviations(df, window_length=DEFAULT_WINDOW_LENGTH, polyorder=DEFAULT_POLYORDER):
    """
    Interpolate missed detections and smooth both arm-angle signals.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per frame with columns ``left_deviation`` and
        ``right_deviation`` in degrees. Frames with no detected pose hold
        NaN.
    window_length : int, optional
        Savitzky-Golay window in frames; must be odd and no longer than the
        clip. Default 9.
    polyorder : int, optional
        Polynomial order; must be less than `window_length`. Default 2.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with added columns ``left_smooth`` and
        ``right_smooth`` (degrees).

    Raises
    ------
    ValueError
        If an arm has no detected value in any frame.

    Notes
    -----
    Interior gaps are filled by linear interpolation between the
    neighbouring detected frames. Gaps at the start or end of the clip are
    filled with the nearest detected value; without this the Savitzky-Golay
    edge fit receives NaN and SciPy raises an error.
    """
    _check_columns(df, ("left_deviation", "right_deviation"))
    out = df.copy()
    for side in ("left", "right"):
        filled = out[f"{side}_deviation"].astype(float).interpolate(limit_direction="both")
        if filled.isna().all():
            raise ValueError(f"No {side}-arm angle was detected in any frame")
        out[f"{side}_smooth"] = savgol_filter(filled.to_numpy(), window_length, polyorder)
    return out


def add_angular_velocity(df):
    """
    Add the angular velocity of each arm, in degrees per second.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `smooth_deviations`, with columns ``time_sec``,
        ``left_smooth`` and ``right_smooth``.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with added columns ``left_velocity`` and
        ``right_velocity`` (deg/s).

    Notes
    -----
    Uses `numpy.gradient`: second-order central differences inside the
    clip and first-order one-sided differences at the two ends, evaluated
    against the actual frame timestamps.
    """
    _check_columns(df, ("time_sec", "left_smooth", "right_smooth"))
    out = df.copy()
    t = out["time_sec"].to_numpy(dtype=float)
    for side in ("left", "right"):
        out[f"{side}_velocity"] = np.gradient(out[f"{side}_smooth"].to_numpy(dtype=float), t)
    return out


def flag_stopped_frames(df, velocity_threshold=VELOCITY_THRESHOLD_DEG_S):
    """
    Mark frames where both arms are effectively stationary.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `add_angular_velocity`.
    velocity_threshold : float, optional
        Speed limit in deg/s that both arms must be under. Default 15.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with a boolean column ``is_stopped``. Frames with a
        NaN velocity are never stopped.
    """
    _check_columns(df, ("left_velocity", "right_velocity"))
    out = df.copy()
    out["is_stopped"] = (out["left_velocity"].abs() < velocity_threshold) & (
        out["right_velocity"].abs() < velocity_threshold
    )
    return out


def find_stopped_segments(df):
    """
    Group consecutive stopped frames into segments.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `flag_stopped_frames`, with columns ``time_sec``,
        ``is_stopped``, ``left_smooth`` and ``right_smooth``.

    Returns
    -------
    pandas.DataFrame
        One row per run of stopped frames, in time order, with columns
        ``start_time`` and ``end_time`` (timestamps of the first and last
        frame, s), ``avg_left_angle`` and ``avg_right_angle`` (deg),
        ``n_frames`` and ``duration`` (``end_time - start_time``, s; a
        single-frame segment has duration 0). Empty if no frame is stopped.
    """
    _check_columns(df, ("time_sec", "is_stopped", "left_smooth", "right_smooth"))
    stopped_mask = df["is_stopped"].astype(bool)
    segment_id = (stopped_mask != stopped_mask.shift()).cumsum()
    stopped = df[stopped_mask]
    segments = stopped.groupby(segment_id[stopped_mask]).agg(
        start_time=("time_sec", "min"),
        end_time=("time_sec", "max"),
        avg_left_angle=("left_smooth", "mean"),
        avg_right_angle=("right_smooth", "mean"),
        n_frames=("time_sec", "count"),
    )
    segments["duration"] = segments["end_time"] - segments["start_time"]
    return segments.reset_index(drop=True)


def merge_segments(segments, gap_tolerance=GAP_TOLERANCE_S):
    """
    Merge stopped segments separated by short gaps.

    Parameters
    ----------
    segments : pandas.DataFrame
        Output of `find_stopped_segments`, sorted by ``start_time``.
    gap_tolerance : float, optional
        Largest gap in seconds, from the end of one segment to the start of
        the next, that is bridged. Default 0.25.

    Returns
    -------
    list of tuple of float
        ``(start_time, end_time)`` pairs in seconds, in time order. Empty if
        `segments` is empty.

    Examples
    --------
    >>> segs = pd.DataFrame({"start_time": [0.0, 1.1, 3.0], "end_time": [1.0, 2.0, 3.5]})
    >>> merge_segments(segs, gap_tolerance=0.25)
    [(0.0, 2.0), (3.0, 3.5)]
    """
    merged = []
    cur_start = cur_end = None
    for _, row in segments.iterrows():
        if cur_start is None:
            cur_start, cur_end = row["start_time"], row["end_time"]
        elif row["start_time"] - cur_end <= gap_tolerance:
            cur_end = row["end_time"]
        else:
            merged.append((float(cur_start), float(cur_end)))
            cur_start, cur_end = row["start_time"], row["end_time"]
    if cur_start is not None:
        merged.append((float(cur_start), float(cur_end)))
    return merged


def average_angle_in_segment(df, start, end):
    """
    Mean deviation over both arms between two timestamps (inclusive).

    Parameters
    ----------
    df : pandas.DataFrame
        Frame table with ``time_sec``, ``left_smooth`` and ``right_smooth``.
    start, end : float
        Segment limits in seconds.

    Returns
    -------
    float
        Mean of the left-arm and right-arm mean smoothed deviations, in
        degrees. NaN if no frame falls in the interval.
    """
    mask = (df["time_sec"] >= start) & (df["time_sec"] <= end)
    return float(df.loc[mask, ["left_smooth", "right_smooth"]].mean().mean())


def build_candidate_holds(df, merged_segments):
    """
    Describe each merged segment as a candidate hold.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame table with ``time_sec``, ``left_smooth`` and ``right_smooth``.
    merged_segments : list of tuple of float
        Output of `merge_segments`.

    Returns
    -------
    pandas.DataFrame
        Columns ``start``, ``end``, ``duration`` (s) and ``avg_angle``
        (deg), one row per segment.
    """
    rows = [
        {
            "start": start,
            "end": end,
            "duration": end - start,
            "avg_angle": average_angle_in_segment(df, start, end),
        }
        for start, end in merged_segments
    ]
    return pd.DataFrame(rows, columns=["start", "end", "duration", "avg_angle"])


def select_main_hold(candidates, max_angle=MAX_CROSS_ANGLE_DEG):
    """
    Choose the hold to judge from the candidate segments.

    Parameters
    ----------
    candidates : pandas.DataFrame
        Output of `build_candidate_holds`.
    max_angle : float, optional
        Candidates with a mean deviation above this (deg) are discarded as
        not being a cross position, e.g. hanging with arms overhead.
        Default 45.

    Returns
    -------
    pandas.Series or None
        The longest remaining candidate, with fields ``start``, ``end``,
        ``duration`` and ``avg_angle``; None if no candidate qualifies.
    """
    if candidates.empty:
        return None
    plausible = candidates[candidates["avg_angle"] <= max_angle]
    if plausible.empty:
        return None
    return plausible.loc[plausible["duration"].idxmax()]


def classify_hold_duration(duration_sec):
    """
    Classify how long the hold lasted.

    Parameters
    ----------
    duration_sec : float
        Hold duration in seconds.

    Returns
    -------
    str
        ``"No deduction"`` for at least 2 s, ``"Medium error"`` for a stop
        shorter than 2 s, and ``"Large error and non-recognition (no stop)"``
        for zero duration.

    Raises
    ------
    ValueError
        If `duration_sec` is NaN or negative.

    Examples
    --------
    >>> classify_hold_duration(1.96)
    'Medium error'
    """
    if duration_sec is None or np.isnan(duration_sec):
        raise ValueError("duration_sec is NaN or None")
    if duration_sec < 0:
        raise ValueError(f"duration_sec must be >= 0, got {duration_sec}")

    if duration_sec >= MIN_HOLD_DURATION_S:
        return "No deduction"
    elif duration_sec > 0:
        return "Medium error"
    else:
        return "Large error and non-recognition (no stop)"


def detect_hold(
    series,
    window_length=None,
    polyorder=DEFAULT_POLYORDER,
    velocity_threshold=VELOCITY_THRESHOLD_DEG_S,
    gap_tolerance=GAP_TOLERANCE_S,
    max_angle=MAX_CROSS_ANGLE_DEG,
):
    """
    Run the full hold-detection chain on a per-frame deviation table.

    Parameters
    ----------
    series : pandas.DataFrame
        One row per frame with at least ``time_sec`` (s),
        ``left_deviation`` and ``right_deviation`` (deg, NaN where no pose
        was detected).
    window_length : int, optional
        Savitzky-Golay window in frames. None (default) chooses one with
        `choose_window_length`.
    polyorder : int, optional
        Savitzky-Golay polynomial order. Default 2.
    velocity_threshold : float, optional
        Stop threshold in deg/s. Default 15.
    gap_tolerance : float, optional
        Largest gap in seconds bridged when merging segments. Default 0.25.
    max_angle : float, optional
        Largest mean deviation in degrees accepted as a cross. Default 45.

    Returns
    -------
    frames : pandas.DataFrame
        `series` with the smoothed angles, velocities and ``is_stopped``
        flag added.
    hold : pandas.Series or None
        The detected hold (``start``, ``end``, ``duration``, ``avg_angle``),
        or None if no plausible hold was found.

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If the clip has fewer than `MIN_WINDOW_LENGTH` frames.
    """
    _check_columns(series, REQUIRED_COLUMNS)
    if window_length is None:
        window_length = choose_window_length(len(series))

    frames = smooth_deviations(series, window_length, polyorder)
    frames = add_angular_velocity(frames)
    frames = flag_stopped_frames(frames, velocity_threshold)

    segments = find_stopped_segments(frames)
    merged = merge_segments(segments, gap_tolerance)
    candidates = build_candidate_holds(frames, merged)
    hold = select_main_hold(candidates, max_angle)
    return frames, hold


def judge_hold(hold):
    """
    Turn a detected hold into a judging summary.

    Parameters
    ----------
    hold : pandas.Series or None
        Output of `select_main_hold` or `detect_hold`.

    Returns
    -------
    dict
        ``start``, ``end``, ``duration``, ``avg_angle``, ``angle_deduction``
        and ``duration_deduction``. If `hold` is None every field is None:
        a missing hold can mean the athlete never stopped or that detection
        failed, and the pipeline cannot tell which, so it does not score it.
    """
    if hold is None:
        return dict.fromkeys(
            ["start", "end", "duration", "avg_angle", "angle_deduction", "duration_deduction"]
        )
    return {
        "start": float(hold["start"]),
        "end": float(hold["end"]),
        "duration": float(hold["duration"]),
        "avg_angle": float(hold["avg_angle"]),
        "angle_deduction": classify_deduction(hold["avg_angle"]),
        "duration_deduction": classify_hold_duration(hold["duration"]),
    }


def timing_errors(hold, manual_start_frame, manual_end_frame, fps):
    """
    Compare a detected hold with a manually annotated one.

    Parameters
    ----------
    hold : pandas.Series
        Detected hold with ``start``, ``end`` and ``duration`` in seconds.
    manual_start_frame, manual_end_frame : int
        Frame indices where a human judged the hold to start and end.
    fps : float
        Video frame rate in frames per second.

    Returns
    -------
    dict of str to float
        ``manual_start``, ``manual_end`` and ``manual_duration`` (s), and
        the signed errors ``start_error``, ``end_error`` and
        ``duration_error`` (s, detected minus manual).
    """
    manual_start = manual_start_frame / fps
    manual_end = manual_end_frame / fps
    manual_duration = manual_end - manual_start
    return {
        "manual_start": manual_start,
        "manual_end": manual_end,
        "manual_duration": manual_duration,
        "start_error": float(hold["start"]) - manual_start,
        "end_error": float(hold["end"]) - manual_end,
        "duration_error": float(hold["duration"]) - manual_duration,
    }
