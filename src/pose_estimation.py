"""
MediaPipe pose estimation and video processing.

This is the only module that needs MediaPipe and OpenCV. It downloads and
loads the Pose Landmarker model, reads images and videos, runs detection on
every frame, and wraps the full pipeline in `process_video`.

Model
-----
MediaPipe Pose Landmarker (BlazePose GHUM, 33 landmarks) is run in IMAGE
mode, so each frame is detected independently with no temporal tracking.
Three model sizes are available; this project uses ``"heavy"`` by default.
"""

__docformat__ = "numpy"

import os
import urllib.request

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.angle_calculation import deviation_from_horizontal, get_arm_points
from src.hold_detection import detect_hold

MODEL_URLS = {
    variant: (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_{variant}/float16/1/pose_landmarker_{variant}.task"
    )
    for variant in ("lite", "full", "heavy")
}
"""Download URLs of the three Pose Landmarker model sizes (version 1)."""


def download_model(variant="heavy", dest_dir="models"):
    """
    Download a Pose Landmarker model file if it is not already present.

    Parameters
    ----------
    variant : {"heavy", "full", "lite"}, optional
        Model size. Heavier models are more accurate and slower.
        Default ``"heavy"``.
    dest_dir : str, optional
        Folder to save the model in. Default ``"models"`` (ignored by git).

    Returns
    -------
    str
        Path to the ``.task`` model file.

    Raises
    ------
    ValueError
        If `variant` is not one of the three model names.
    """
    if variant not in MODEL_URLS:
        raise ValueError(f"variant must be one of {sorted(MODEL_URLS)}, got {variant!r}")
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"pose_landmarker_{variant}.task")
    if not os.path.exists(path):
        urllib.request.urlretrieve(MODEL_URLS[variant], path)
    return path


def create_detector(model_path, output_segmentation_masks=False):
    """
    Create a MediaPipe Pose Landmarker for single images.

    Parameters
    ----------
    model_path : str
        Path to a ``.task`` model file (see `download_model`).
    output_segmentation_masks : bool, optional
        Also return a person segmentation mask. The pipeline does not use
        it, so the default is False. Default False.

    Returns
    -------
    mediapipe.tasks.python.vision.PoseLandmarker
        Detector in IMAGE running mode, detecting at most one person.
    """
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        output_segmentation_masks=output_segmentation_masks,
    )
    return vision.PoseLandmarker.create_from_options(options)


def load_image_rgb(path):
    """
    Read an image file as an RGB array.

    Parameters
    ----------
    path : str
        Path to an image file OpenCV can read (e.g. JPEG, PNG; not HEIC).

    Returns
    -------
    numpy.ndarray
        Image of shape ``(height, width, 3)``, dtype uint8, RGB order.

    Raises
    ------
    FileNotFoundError
        If the file is missing or cannot be decoded.
    """
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_landmarks(detector, frame_rgb):
    """
    Run pose detection on one RGB image.

    Parameters
    ----------
    detector : mediapipe.tasks.python.vision.PoseLandmarker
        Detector from `create_detector`.
    frame_rgb : numpy.ndarray
        RGB image, shape ``(height, width, 3)``, dtype uint8.

    Returns
    -------
    list or None
        The 33 normalised landmarks of the first detected person, or None
        if no person was found.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)
    if not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]


def read_video_info(video_path):
    """
    Read basic properties of a video file.

    Parameters
    ----------
    video_path : str
        Path to a video file OpenCV can read (e.g. .mp4, .mov).

    Returns
    -------
    dict
        ``fps`` (frames/s), ``n_frames``, ``duration_s`` (s), ``width`` and
        ``height`` (pixels). ``n_frames`` comes from the file header and can
        differ slightly from the number of frames actually decoded.

    Raises
    ------
    OSError
        If the file cannot be opened or reports a frame rate of zero.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0:
        raise OSError(f"Video reports an invalid frame rate ({fps}): {video_path}")
    return {
        "fps": fps,
        "n_frames": n_frames,
        "duration_s": n_frames / fps,
        "width": width,
        "height": height,
    }


def read_all_frames(video_path):
    """
    Decode every frame of a video into memory as RGB arrays.

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Returns
    -------
    list of numpy.ndarray
        One RGB array per frame. A 5 s 1080p clip at 30 fps needs about
        1 GB, so use this only for short clips.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def read_frame_at_time(video_path, time_sec):
    """
    Read the single frame nearest a given time.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    time_sec : float
        Time from the start of the clip in seconds.

    Returns
    -------
    numpy.ndarray or None
        RGB frame, or None if the frame could not be read.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_sec * fps))
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_deviation_series(video_path, detector):
    """
    Measure both arms' deviation from horizontal on every frame of a video.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    detector : mediapipe.tasks.python.vision.PoseLandmarker
        Detector from `create_detector`.

    Returns
    -------
    series : pandas.DataFrame
        One row per decoded frame with columns ``frame`` (index),
        ``time_sec`` (s, ``frame / fps``), ``left_deviation`` and
        ``right_deviation`` (deg, NaN where no pose was found) and
        ``detected`` (bool).
    fps : float
        Frame rate used for the timestamps.

    Raises
    ------
    OSError
        If the video cannot be opened (see `read_video_info`).
    """
    fps = read_video_info(video_path)["fps"]
    cap = cv2.VideoCapture(video_path)
    rows = []
    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        landmarks = detect_landmarks(detector, frame_rgb)

        row = {"frame": frame_index, "time_sec": frame_index / fps}
        if landmarks is not None:
            pts = get_arm_points(landmarks, w, h)
            row["left_deviation"] = deviation_from_horizontal(pts["left_shoulder"], pts["left_wrist"])
            row["right_deviation"] = deviation_from_horizontal(pts["right_shoulder"], pts["right_wrist"])
            row["detected"] = True
        else:
            row["left_deviation"] = float("nan")
            row["right_deviation"] = float("nan")
            row["detected"] = False
        rows.append(row)
        frame_index += 1
    cap.release()

    columns = ["frame", "time_sec", "left_deviation", "right_deviation", "detected"]
    return pd.DataFrame(rows, columns=columns), fps


def process_video(video_path, detector, **hold_kwargs):
    """
    Run the complete judging pipeline on one video.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    detector : mediapipe.tasks.python.vision.PoseLandmarker
        Detector from `create_detector`.
    **hold_kwargs
        Optional settings passed to `src.hold_detection.detect_hold`, e.g.
        ``velocity_threshold=20``.

    Returns
    -------
    frames : pandas.DataFrame
        Per-frame table with raw and smoothed angles, velocities and the
        ``is_stopped`` flag.
    hold : pandas.Series or None
        The detected hold, or None if no plausible hold was found. Pass it
        to `src.hold_detection.judge_hold` for the deduction categories.

    Examples
    --------
    >>> detector = create_detector(download_model("heavy"))   # doctest: +SKIP
    >>> frames, hold = process_video("clip.mp4", detector)    # doctest: +SKIP
    """
    series, _ = extract_deviation_series(video_path, detector)
    return detect_hold(series, **hold_kwargs)
