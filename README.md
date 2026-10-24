# iron-cross-judging

Automated judging of the gymnastics **iron cross** on still rings from ordinary video. The pipeline uses MediaPipe pose estimation to measure how far each arm deviates from horizontal, finds the held part of the element, and classifies the arm angle and the hold duration into the deduction categories of the FIG Men's Artistic Gymnastics Code of Points (2025–2028).

MMA3001 Numerical Methods and Machine Learning, individual project. Author: Joshua Walker (Monash University).

---

## Contents

1. [Engineering problem](#engineering-problem)
2. [How it works](#how-it-works)
3. [Inputs](#inputs)
4. [Outputs](#outputs)
5. [Repository structure](#repository-structure)
6. [Getting started](#getting-started)
7. [Usage](#usage)
8. [Data](#data)
9. [Validation and current results](#validation-and-current-results)
10. [Testing](#testing)
11. [Documentation](#documentation)
12. [Known limitations](#known-limitations)
13. [AI use](#ai-use)
14. [References](#references)
15. [License](#license)

---

## Engineering problem

**Context.** The iron cross is a strength hold on still rings: the gymnast supports their body weight with the arms held straight out to the sides at shoulder height. Under the 2025–2028 Code of Points, execution judges deduct for two things in a strength hold:

| Criterion | Requirement | Deduction |
|---|---|---|
| Arm angle | Arms horizontal | 0 to 5° none; >5 to 20° small (0.1); >20 to 45° medium (0.3); >45° large (0.5) and the element is not recognised |
| Hold time | Held still for 2 s | Stop shorter than 2 s: medium (0.3); no stop: large (0.5) and not recognised |

**The problem.** Judges assess both criteria by eye, in real time, from wherever they are seated. Telling a 5° deviation from a 7° one by eye is hard, and so is timing two seconds exactly. Hold-time deductions alone made up roughly 18% of all deductions in one study of junior rings routines [5]. Commercial judging-support systems exist but rely on dedicated multi-sensor hardware.

**Intended use.** A low-cost tool that gives gymnasts and coaches objective feedback on cross angle and hold time from a phone or broadcast video, and that could support judge training. It is **not** intended for official scoring.

**Scope.**
- One element (iron cross), one athlete in frame, camera roughly front-on.
- 2D measurement in the image plane; no depth.
- Validated so far on 1 manually annotated clip and 9 clips without frame-level annotation (2 personal, 7 from the 2025 World Championships).

---

## How it works

```
video ──► pose estimation ──► arm angle ──► smoothing ──► angular velocity ──► stop detection
          (MediaPipe,          (shoulder→    (Savitzky-     (central              (|ω| < 15°/s,
           33 landmarks)        wrist vs      Golay)         differences)          both arms)
                                horizontal)                                            │
                                                                                       ▼
deductions ◄── classification ◄── hold selection ◄── segment merging ◄── stopped segments
               (CoP angle and      (longest with       (gaps ≤ 0.25 s)
                time bands)         mean angle ≤ 45°)
```

1. **Pose estimation.** MediaPipe Pose Landmarker (BlazePose GHUM [2, 3]) returns 33 normalised body landmarks for every frame. Each frame is processed independently (IMAGE mode).
2. **Arm angle.** For each arm, the angle of the shoulder-to-wrist vector is folded into a deviation from horizontal in [0°, 90°]. Frames with no detected pose are recorded as NaN.
3. **Smoothing.** Missed frames are filled by linear interpolation, then each arm's signal is smoothed with a Savitzky–Golay filter [4]. Unlike a moving average of the same length, it fits a local polynomial, so it removes landmark jitter while keeping the sharp corners where the hold starts and ends.
4. **Angular velocity.** `numpy.gradient` (second-order central differences) against the real frame timestamps.
5. **Stop detection.** A frame is "stopped" when both arms move slower than the velocity threshold.
6. **Segment merging.** Runs of stopped frames separated by short gaps (a tremor, a missed detection) are joined.
7. **Hold selection.** Segments whose mean angle exceeds 45° (for example hanging with arms overhead) cannot be a cross and are discarded; the longest remaining segment is the hold.
8. **Classification.** The hold's mean angle over both arms and its duration are mapped to the bands in the table above.

### Parameters

| Parameter | Value | Where | Basis |
|---|---|---|---|
| Pose model | `heavy` (float16, v1) | `pose_estimation.download_model` | Most accurate of the three MediaPipe sizes; `lite` and `full` are available for runtime comparison |
| Smoothing window | 9 frames (≈0.3 s at 30 fps) | `hold_detection.DEFAULT_WINDOW_LENGTH` | Chosen by inspecting HPCComp.mov; shortened automatically for clips under 9 frames |
| Polynomial order | 2 | `hold_detection.DEFAULT_POLYORDER` | Chosen by inspection |
| Stop threshold | 15 °/s | `hold_detection.VELOCITY_THRESHOLD_DEG_S` | Chosen by inspecting the velocity plot of HPCComp.mov |
| Gap tolerance | 0.25 s | `hold_detection.GAP_TOLERANCE_S` | Chosen by inspection |
| Max cross angle | 45° | `hold_detection.MAX_CROSS_ANGLE_DEG` | Code of Points non-recognition limit |
| Required hold | 2.0 s | `hold_detection.MIN_HOLD_DURATION_S` | Code of Points |

All tunable values are arguments of `detect_hold` / `process_video`, so a sensitivity analysis can vary them without editing the code.

---

## Inputs

| Input | Format | Units / domain | Notes |
|---|---|---|---|
| Video clip | Any file OpenCV can decode (`.mp4`, `.mov`) | Frame rate read from the file (Hz) | One athlete, camera roughly front-on, both arms and the hips visible |
| Reference photo | JPEG or PNG (not HEIC) | Pixels | Front-on still of a cross; used only by the perspective correction |
| Manual annotation | Two integer frame indices | Frames | First and last frame of the hold, chosen by eye; used only for validation |

### Invalid, missing or unsupported inputs

| Situation | Behaviour |
|---|---|
| Video or image cannot be opened | `OSError` / `FileNotFoundError` with the path |
| Video reports a frame rate of 0 | `OSError` |
| No pose found in a frame | Angle stored as NaN; interior gaps are linearly interpolated, gaps at the start or end take the nearest detected value |
| No pose found in any frame | `ValueError` |
| Fewer than 5 frames | `ValueError` (too short to smooth) |
| No stop, or no stop with a plausible cross angle | `hold` is `None`; `judge_hold` returns all fields as `None` |
| NaN or negative angle or duration passed to a classifier | `ValueError` (never silently scored) |
| More than one person in frame | Only the first detected person is used; results are unreliable |

---

## Outputs

**Per-frame table** (`frames`, a pandas DataFrame, one row per frame):

| Column | Units | Meaning |
|---|---|---|
| `frame` | – | Frame index |
| `time_sec` | s | `frame / fps` |
| `left_deviation`, `right_deviation` | ° | Raw deviation from horizontal (NaN if no pose) |
| `detected` | bool | Whether a pose was found |
| `left_smooth`, `right_smooth` | ° | Smoothed deviation |
| `left_velocity`, `right_velocity` | °/s | Angular velocity |
| `is_stopped` | bool | Both arms below the stop threshold |

**Judging summary** (`judge_hold(hold)`, a dict):

| Key | Units | Meaning |
|---|---|---|
| `start`, `end` | s | Detected start and end of the hold |
| `duration` | s | `end − start` |
| `avg_angle` | ° | Mean smoothed deviation over the hold, averaged over both arms |
| `angle_deduction` | – | `No deduction`, `Small error`, `Medium error` or `Large error` |
| `duration_deduction` | – | `No deduction`, `Medium error` or `Large error and non-recognition (no stop)` |

`angle_calculation.DEDUCTION_POINTS` converts a category to points (0.0, 0.1, 0.3, 0.5).

---

## Repository structure

```
iron-cross-judging/
├── README.md                  this file
├── LICENSE                    MIT
├── NOTES.md                   development log and findings
├── requirements.txt           Python dependencies
├── pytest.ini                 test configuration
├── src/                       the pipeline, as an importable package
│   ├── __init__.py            package overview (pdoc landing page)
│   ├── angle_calculation.py   landmarks → arm deviation → angle deduction
│   ├── hold_detection.py      smoothing, velocity, stop detection, hold selection
│   ├── perspective.py         experimental camera-yaw correction
│   └── pose_estimation.py     MediaPipe model, video reading, process_video()
├── tests/                     pytest unit, integration and regression tests
├── notebooks/
│   └── main_pipeline.ipynb    end-to-end analysis, plots and validation
├── docs/                      generated HTML API docs and test report
└── data/sample/               placeholder; raw data is not distributed (see Data)
```

Pose model files are downloaded into `models/` on first run and are ignored by git.

---

## Getting started

### Google Colab (how the results were produced)

1. Open `notebooks/main_pipeline.ipynb` in Colab (*File → Open notebook → GitHub*, search `Joshua-Walker978/iron-cross-judging`).
2. Put the video clips in a Google Drive folder and set `DATA_DIR` in section 0 of the notebook.
3. *Runtime → Run all.* Section 0 clones the repo, installs `requirements.txt` and downloads the pose model.

### Local

Requires Python 3.10 or later.

```bash
git clone https://github.com/Joshua-Walker978/iron-cross-judging.git
cd iron-cross-judging
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                             # should report all tests passing
```

The notebook's first two cells are Colab-specific (Drive mount and clone); skip them when running locally and set `DATA_DIR` to a local folder. The notebook prints the exact library versions it ran with, for reproducibility.

---

## Usage

From the repository root:

```python
from src.pose_estimation import create_detector, download_model, process_video
from src.hold_detection import judge_hold

detector = create_detector(download_model("heavy"))
frames, hold = process_video("my_cross.mp4", detector)
print(judge_hold(hold))
# e.g. for HPCComp.mov:
# {'start': 1.26, 'end': 3.23, 'duration': 1.96, 'avg_angle': 28.1,
#  'angle_deduction': 'Medium error', 'duration_deduction': 'Medium error'}
```

Change a setting without editing code:

```python
frames, hold = process_video("my_cross.mp4", detector, velocity_threshold=20, gap_tolerance=0.3)
```

---

## Data

The raw videos are **not** in this repository.

- **Personal recordings** (`HPCComp.mov`, `JoshCross1/2`, `IMG_2471.jpg`): recorded of myself with my consent, kept private.
- **2025 World Championships clips** (`2025Worlds1` to `7`): short excerpts of broadcast footage used for research analysis only. They are copyrighted by the rights holder and cannot be redistributed.

Expected layout of the data folder:

```
<DATA_DIR>/
├── IMG_2471.jpg       front-on reference photo
├── HPCComp.mov        annotated validation clip
├── JoshCross1.mp4
├── JoshCross2.mp4
└── 2025Worlds1.mp4 … 2025Worlds7.mp4
```

`.gitignore` blocks `*.mp4`, `*.mov` and `*.HEIC` so raw footage cannot be committed by accident.

---

## Validation and current results

Results below are from the notebook run in October 2026. Re-running the notebook regenerates them.

### 1. Code verification (does the code do what it is meant to?)

- **Manufactured solutions:** synthetic angle signals with a hold of known start, end and angle (with and without noise and missed frames) are recovered within 0.25 s and 1° (`tests/test_hold_detection.py`).
- **Exact geometry:** the angle maths is checked against known angles, mirror symmetry and band edges; the perspective correction recovers a known yaw and angle exactly on a synthetic rotated body (`tests/test_perspective.py`).
- **Integration:** the real video-reading loop is run on a generated video with a stand-in detector, checking timestamps, missed-frame handling and the end-to-end result (`tests/test_pose_estimation.py`).
- **Regression:** the `src/` modules reproduce the original notebook implementation to 1e-12 on 200 random signals (`tests/test_matches_original_notebook.py`).

### 2. Hold timing against manual annotation (HPCComp.mov)

The hold's first and last frames were marked by eye, frame by frame.

| | Start (s) | End (s) | Duration (s) |
|---|---|---|---|
| Manual | 1.56 | 3.43 | 1.86 |
| Algorithm | 1.26 | 3.23 | 1.96 |
| Error | −0.30 | −0.20 | +0.10 |

The detected hold starts about 9 frames early and ends about 6 frames early. The duration error (+0.10 s) is small, but errors of 0.2–0.3 s at either end matter when a hold is close to the 2 s requirement.

### 3. Batch evaluation (9 clips)

| Clip | Duration (s) | Mean angle (°) | Angle deduction | Time deduction |
|---|---|---|---|---|
| JoshCross1 | 2.24 | 16.8 | Small | None |
| JoshCross2 | 1.90 | 18.1 | Small | Medium |
| 2025Worlds1 | 1.97 | 9.1 | Small | Medium |
| 2025Worlds2 | 3.20 | 10.4 | Small | None |
| 2025Worlds3 | 2.03 | 14.2 | Small | None |
| 2025Worlds4 | 1.57 | 11.2 | Small | Medium |
| 2025Worlds5 | 2.03 | 14.4 | Small | None |
| 2025Worlds6 | 4.83 | 13.3 | Small | None |
| 2025Worlds7 | 2.90 | 10.2 | Small | None |

A hold was detected in every clip. All seven World Championship crosses measured 9–14°, although by my own visual judgement every one should be within 5° (no deduction). This consistent overestimate is attributed mainly to camera perspective: broadcast cameras are rarely square to the athlete.

### 4. Perspective correction (experimental, under re-evaluation)

A body-proportion yaw correction (`src/perspective.py`) was built to address the bias. Earlier runs showed little improvement, but they were affected by a bug in which the reference ratio was computed from the wrong landmarks; this is fixed and the evaluation is being re-run. See NOTES.md for the analysis of why apparatus-based calibration was not pursued.

### What the validation cannot establish

- Angle accuracy against true 3D ground truth: no motion-capture or calibrated reference was available, so "true" angles rest on visual judgement.
- Generalisation: timing was validated against manual annotation on one clip only.
- Agreement with official judges: no official execution scores were available for the clips.

---

## Testing

```bash
pytest                                                          # run all tests
pytest -v --html=docs/test_report.html --self-contained-html    # with an HTML report
```

Only `tests/test_pose_estimation.py` needs MediaPipe and OpenCV; it is skipped automatically if they are not installed. The latest report is `docs/test_report.html`.

---

## Documentation

HTML API documentation is generated from the NumPy-style docstrings with [pdoc](https://pdoc.dev):

```bash
pdoc src -o docs
```

Open `docs/index.html` in a browser. To publish it, enable GitHub Pages (*Settings → Pages → Deploy from a branch → main, /docs*); the docs and test report are then available at `https://joshua-walker978.github.io/iron-cross-judging/`.

---

## Known limitations

- **Camera perspective.** Angles are measured in 2D. Any yaw between camera and athlete makes the arms look steeper, which inflates the deviation (see Validation 3).
- **No minimum stop.** An arm that decelerates into the cross and reverses without pausing is still briefly below the 15 °/s threshold, so a synthetic "touch and go" with no pause at all is reported as a 0.2 s hold (medium error) where the Code of Points would give no stop (large error, not recognised). Only a perfectly sharp reversal gives zero duration.
- **Parameters tuned on one clip.** Thresholds were chosen by inspecting HPCComp.mov; their sensitivity has not yet been quantified.
- **Folded angle.** Arms raised above and dropped below horizontal by the same amount give the same deviation, which matches how the deduction is applied but loses direction information.
- **One person.** Coaches or spotters in frame can be detected instead of the athlete.
- **Elbow bend and body position** are not assessed, although they are also judged.

---

## AI use

> **TODO (Joshua):** complete this section in your own words before submission. The brief requires: which tools, what for, approximate level of contribution, why, how output was checked, which decisions stayed yours, errors or unhelpful suggestions, and the effect on your learning.

Starting point:

- **Claude (Anthropic)** was used to identify redundant notebook cells; to move the notebook code into the documented `src/` modules; to write the docstrings, the unit and regression tests, and a first draft of this README; and to draft presentation slides. It also flagged a bug in the perspective-correction reference and a crash on clips whose first frame has no detection.
- **Checking:** the restructured code was verified against the original implementation by `tests/test_matches_original_notebook.py`, and section 3 of the notebook compares the new pipeline's result on HPCComp.mov with the previously recorded result. *(add your own checks)*
- **Decisions that remained mine:** *(e.g. project choice, the deduction bands, parameter values, manual annotation, interpretation of results)*

---

## References

1. Fédération Internationale de Gymnastique (2024). *2025–2028 Code of Points, Men's Artistic Gymnastics.* https://www.gymnastics.sport/publicdir/rules/files/en_1.1%20-%20MAG%20CoP%202025-2028.pdf
2. Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F. & Grundmann, M. (2020). *BlazePose: On-device Real-time Body Pose tracking.* arXiv:2006.10204.
3. Google. *MediaPipe Pose Landmarker guide.* https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
4. Savitzky, A. & Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, 36(8), 1627–1639.
5. Aarts, M., Wolfs, L. & Pluk, A. (2016). Making Every Second Count: Implementation of the Smart Rings Jury Tool to Evaluate Hold Time during the Men's Artistic Gymnastics European Championships 2016. *icSPORTS 2016*, 29–30. https://www.scitepress.org/Papers/2016/60847/60847.pdf

---

## License

MIT; see [LICENSE](LICENSE).
