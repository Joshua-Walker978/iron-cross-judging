"""Regression test: the src/ modules reproduce the original notebook logic.

`original_process_video_tail` is a frozen copy of the post-detection part of
`process_video` as it was written in the notebook before the code moved into
src/. Both versions are run on random synthetic angle signals and must agree
to 1e-12 on every intermediate result and on the detected hold.

The original crashes when the first frame has no detection (SciPy rejects
NaN at the filter edge); the new code fills that gap instead, so those
signals are skipped here and covered by
test_hold_detection.test_smoothing_handles_missed_first_and_last_frames.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.hold_detection import detect_hold


def original_process_video_tail(results_list):
    """Frozen copy of the original notebook code (do not edit)."""
    video_df = pd.DataFrame(results_list)
    window_length = min(9, len(video_df) - (1 - len(video_df) % 2))
    if window_length < 5:
        window_length = 5
    video_df["left_smooth"] = savgol_filter(video_df["left_deviation"].interpolate(), window_length, 2)
    video_df["right_smooth"] = savgol_filter(video_df["right_deviation"].interpolate(), window_length, 2)
    video_df["left_velocity"] = np.gradient(video_df["left_smooth"], video_df["time_sec"])
    video_df["right_velocity"] = np.gradient(video_df["right_smooth"], video_df["time_sec"])
    velocity_threshold = 15
    video_df["is_stopped"] = (video_df["left_velocity"].abs() < velocity_threshold) & (
        video_df["right_velocity"].abs() < velocity_threshold
    )
    video_df["segment_id"] = (video_df["is_stopped"] != video_df["is_stopped"].shift()).cumsum()
    stopped = video_df[video_df["is_stopped"] == True].groupby("segment_id").agg(  # noqa: E712
        start_time=("time_sec", "min"), end_time=("time_sec", "max")
    )
    gap_tolerance = 0.25
    merged = []
    cur_start = cur_end = None
    for _, row in stopped.iterrows():
        if cur_start is None:
            cur_start, cur_end = row["start_time"], row["end_time"]
        elif row["start_time"] - cur_end <= gap_tolerance:
            cur_end = row["end_time"]
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = row["start_time"], row["end_time"]
    if cur_start is not None:
        merged.append((cur_start, cur_end))
    candidates = []
    for start, end in merged:
        mask = (video_df["time_sec"] >= start) & (video_df["time_sec"] <= end)
        avg_angle = video_df.loc[mask, ["left_smooth", "right_smooth"]].mean().mean()
        candidates.append({"start": start, "end": end, "duration": end - start, "avg_angle": avg_angle})
    candidates_df = pd.DataFrame(candidates)
    if candidates_df.empty:  # the original raised KeyError here; treat as "no hold"
        return video_df, None
    plausible = candidates_df[candidates_df["avg_angle"] <= 45]
    if len(plausible) == 0:
        return video_df, None
    return video_df, plausible.loc[plausible["duration"].idxmax()]


def random_rows(rng):
    """A random walk of arm angles with jitter and random missed detections."""
    fps = rng.choice([25.0, 29.97, 30.063965884861407, 60.0])
    n = int(rng.integers(5, 400))
    base = np.clip(np.cumsum(rng.normal(0, rng.choice([0.3, 2.0, 6.0]), n)) + rng.uniform(0, 90), 0, 90)
    rows = []
    for i in range(n):
        missed = rng.random() < 0.05
        rows.append(
            {
                "frame": i,
                "time_sec": i / fps,
                "left_deviation": np.nan if missed else base[i] + rng.normal(0, 0.5),
                "right_deviation": np.nan if missed else base[i] + 1 + rng.normal(0, 0.5),
                "detected": not missed,
            }
        )
    return rows


def test_new_code_matches_original_notebook():
    rng = np.random.default_rng(42)
    compared = holds = 0
    for _ in range(200):
        rows = random_rows(rng)
        if np.isnan(rows[0]["left_deviation"]):
            continue  # original crashes on a missed first frame (see module docstring)
        old_df, old_hold = original_process_video_tail(rows)
        new_df, new_hold = detect_hold(pd.DataFrame(rows))

        for col in ["left_smooth", "right_smooth", "left_velocity", "right_velocity"]:
            np.testing.assert_allclose(new_df[col], old_df[col], rtol=0, atol=1e-12)
        assert (new_df["is_stopped"].to_numpy() == old_df["is_stopped"].to_numpy()).all()
        if old_hold is None:
            assert new_hold is None
        else:
            holds += 1
            for key in ["start", "end", "duration", "avg_angle"]:
                assert abs(new_hold[key] - old_hold[key]) < 1e-12
        compared += 1

    assert compared >= 150 and holds >= 50  # make sure the test actually exercised both paths
