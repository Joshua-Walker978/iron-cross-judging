"""Integration tests for src.pose_estimation.

These run the real video-reading loop on a small synthetic video, but replace
the MediaPipe model with a stand-in detector that returns known landmarks.
That isolates our own code (frame loop, timestamps, angle maths, missed
detections, hold detection) from the accuracy of the pose model, which is
validated separately against manual annotation in the notebook.
"""

from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from src.angle_calculation import LEFT_SHOULDER, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_WRIST  # noqa: E402
from src.pose_estimation import (  # noqa: E402
    MODEL_URLS,
    download_model,
    extract_deviation_series,
    load_image_rgb,
    process_video,
    read_video_info,
)

FPS = 30.0


class FakeDetector:
    """Returns landmarks for a known arm angle per frame; misses chosen frames."""

    def __init__(self, angle_fn, missed=()):
        self.angle_fn = angle_fn
        self.missed = set(missed)
        self.calls = 0

    def detect(self, mp_image):
        i = self.calls
        self.calls += 1
        if i in self.missed:
            return SimpleNamespace(pose_landmarks=[])
        rad = np.radians(self.angle_fn(i / FPS))
        dx, dy = 0.2 * np.cos(rad), 0.2 * np.sin(rad)
        lms = [SimpleNamespace(x=0.5, y=0.5) for _ in range(33)]
        lms[LEFT_SHOULDER] = SimpleNamespace(x=0.55, y=0.3)
        lms[RIGHT_SHOULDER] = SimpleNamespace(x=0.45, y=0.3)
        lms[LEFT_WRIST] = SimpleNamespace(x=0.55 + dx, y=0.3 + dy)
        lms[RIGHT_WRIST] = SimpleNamespace(x=0.45 - dx, y=0.3 + dy)
        return SimpleNamespace(pose_landmarks=[lms])


@pytest.fixture
def video_path(tmp_path):
    """Write a 6 s, 64x64 px synthetic video (content is irrelevant to the fake detector)."""
    path = str(tmp_path / "synthetic.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 64))
    for i in range(180):
        writer.write(np.full((64, 64, 3), i % 255, dtype=np.uint8))
    writer.release()
    return path


def profile(t):
    """Square image, so normalised and pixel angles agree: hang, cross 2-4.5 s, hang."""
    if t < 1.0 or t >= 5.5:
        return 85.0
    if t < 2.0:
        return 85.0 - 81.0 * (t - 1.0)
    if t <= 4.5:
        return 4.0
    return 4.0 + 81.0 * (t - 4.5)


def test_read_video_info(video_path):
    info = read_video_info(video_path)
    assert info["fps"] == pytest.approx(FPS)
    assert info["n_frames"] == 180
    assert info["width"] == 64 and info["height"] == 64


def test_missing_files_raise():
    with pytest.raises(OSError):
        read_video_info("does_not_exist.mp4")
    with pytest.raises(FileNotFoundError):
        load_image_rgb("does_not_exist.jpg")


def test_extract_deviation_series(video_path):
    detector = FakeDetector(profile, missed=[100, 101])
    series, fps = extract_deviation_series(video_path, detector)
    assert fps == pytest.approx(FPS)
    assert len(series) == detector.calls == 180
    assert list(series.columns) == ["frame", "time_sec", "left_deviation", "right_deviation", "detected"]
    np.testing.assert_allclose(series["time_sec"], np.arange(180) / FPS)
    assert not series.loc[[100, 101], "detected"].any()
    assert series.loc[[100, 101], "left_deviation"].isna().all()
    assert series.loc[90, "left_deviation"] == pytest.approx(4.0, abs=1e-6)
    assert series.loc[90, "right_deviation"] == pytest.approx(4.0, abs=1e-6)


def test_process_video_end_to_end(video_path):
    frames, hold = process_video(video_path, FakeDetector(profile, missed=[100, 101]))
    assert len(frames) == 180
    assert hold["start"] == pytest.approx(2.0, abs=0.2)
    assert hold["end"] == pytest.approx(4.5, abs=0.2)
    assert hold["avg_angle"] == pytest.approx(4.0, abs=1.0)


def test_download_model_rejects_unknown_variant(tmp_path):
    with pytest.raises(ValueError):
        download_model("medium", dest_dir=str(tmp_path))
    assert set(MODEL_URLS) == {"lite", "full", "heavy"}
