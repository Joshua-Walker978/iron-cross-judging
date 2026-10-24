# iron-cross-judging

Automated judging of the gymnastics iron cross from video. Given a clip of an athlete on still rings, the pipeline finds the held part of the cross and reports its duration and mean arm angle, each classified into a deduction category from the FIG Men's Artistic Gymnastics Code of Points (2025–2028).

MMA3001 Numerical Methods and Machine Learning, individual project. Joshua Walker, Monash University.

The engineering background, validation and discussion are in the project report, submitted separately. This README covers how to set up, run and reproduce the work.

**Contents:** [Quick start](#quick-start-google-colab) · [Run locally](#run-locally) · [Environment](#environment-and-dependencies) · [Structure](#repository-structure) · [Reproducing the results](#reproducing-the-results) · [Using the package](#using-the-package) · [Data](#data) · [Tests](#tests) · [Documentation](#documentation) · [Development workflow](#development-workflow) · [Troubleshooting](#troubleshooting)

---

## Quick start (Google Colab)

The results were produced in Google Colab, and this is the supported way to run the project.

1. In Colab, choose *File → Open notebook → GitHub*, enter `Joshua-Walker978/iron-cross-judging` and open `notebooks/main_pipeline.ipynb`.
2. Put the video clips in a Google Drive folder (see [Data](#data)) and set `DATA_DIR` in section 0 of the notebook to that folder.
3. Choose *Runtime → Run all* and allow Google Drive access when asked.

Section 0 clones this repository, installs `requirements.txt` and downloads the pose model, so nothing needs installing beforehand.

---

## Run locally

```bash
git clone https://github.com/Joshua-Walker978/iron-cross-judging.git
cd iron-cross-judging
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                             # expected result: 84 passed
```

To run the notebook locally, also `pip install notebook`. The first two cells are Colab-only (Drive mount and clone): skip them, and run `%cd ..` first so the working directory is the repository root, which the `from src...` imports need.

---

## Environment and dependencies

| Package | Used for |
|---|---|
| `mediapipe` | Pose landmark detection (Pose Landmarker) |
| `opencv-python` | Reading video and images, drawing overlays |
| `numpy` | Geometry and numerical differentiation |
| `pandas` | Per-frame result tables |
| `scipy` | Savitzky–Golay smoothing filter |
| `matplotlib` | Plots |
| `ipywidgets` | Frame-by-frame scrubber in the notebook |
| `pytest`, `pytest-html` | Tests and the HTML test report |
| `pdoc` | HTML API documentation |

All are listed with minimum versions in `requirements.txt`.

- **Python:** 3.10 or later; developed and tested on 3.12.
- **Known-good versions:** all tests pass with mediapipe 0.10.33, OpenCV 4.13, NumPy 2.4, pandas 3.0 and SciPy 1.17. Every notebook run prints the versions it used (section 0), so results can be traced to an environment.
- **Pose model:** MediaPipe Pose Landmarker `heavy` (float16, version 1). It is downloaded automatically into `models/` on first use, so the first run needs internet access. `models/` is ignored by git.
- **Hardware:** CPU only; no GPU needed.

---

## Repository structure

```
iron-cross-judging/
├── README.md                  setup, usage and workflow (this file)
├── LICENSE                    MIT
├── NOTES.md                   development log: findings and design decisions
├── requirements.txt           Python dependencies
├── pytest.ini                 test configuration (run pytest from the repo root)
├── src/                       the pipeline as an importable package
│   ├── __init__.py            package overview (landing page of the API docs)
│   ├── angle_calculation.py   landmarks → arm angle → angle deduction
│   ├── hold_detection.py      smoothing, velocity, stop detection, hold selection
│   ├── perspective.py         experimental camera-yaw correction
│   └── pose_estimation.py     MediaPipe model, video reading, process_video()
├── tests/                     one test file per module, plus a regression test
├── notebooks/
│   └── main_pipeline.ipynb    end-to-end analysis, plots and validation
├── docs/                      generated HTML API docs and test report
└── data/sample/               placeholder; raw data is not stored here (see Data)
```

**Conventions.**
- Code that does the processing lives in `src/`. The notebook only loads data, calls those functions and plots the results.
- Module, function and file names use `snake_case`, and each module has one responsibility.
- Test files mirror module names (`test_<module>.py`).
- Constants are `UPPER_CASE` with the unit in the name where there is one, e.g. `VELOCITY_THRESHOLD_DEG_S`, `GAP_TOLERANCE_S`.

---

## Reproducing the results

`notebooks/main_pipeline.ipynb` reproduces every result in the report. Its sections are:

| Section | What it does | Output |
|---|---|---|
| 0. Setup | Mounts Drive, clones the repo, installs dependencies, loads the model | Library versions |
| 1. Single photo | Pose detection and angle maths on `IMG_2471.jpg` | Labelled photo, angle of each arm |
| 2. One video, step by step | Runs `HPCComp.mov` through each processing stage | Raw, smoothed and velocity plots; deduction summary |
| 3. Validation: hold timing | Compares the detected hold with a manual frame annotation | Timing errors; regression checks |
| 4. Batch evaluation | Runs the full pipeline on all 9 clips | Results table, run time per clip, mid-hold frames |
| 5. Perspective correction | Experimental camera-yaw correction | Raw vs corrected angles |
| 6. Tests and documentation | Runs `pytest` and `pdoc` | `docs/test_report.html`, `docs/*.html` |
| 7. Save to GitHub | Commits and pushes (only when a commit message is set) | – |

**Checking a run reproduced correctly.** Section 3 asserts that the one-call pipeline gives exactly the same hold as the step-by-step version. It also prints the hold found for `HPCComp.mov` beside the previously recorded result (1.26 s to 3.23 s, mean angle 28.1°). Small differences can come from a different MediaPipe version.

---

## Using the package

From the repository root:

```python
from src.pose_estimation import create_detector, download_model, process_video
from src.hold_detection import judge_hold

detector = create_detector(download_model("heavy"))
frames, hold = process_video("path/to/clip.mp4", detector)
print(judge_hold(hold))
```

`judge_hold` returns the hold's `start`, `end` and `duration` (s), its `avg_angle` (°), and the `angle_deduction` and `duration_deduction` categories. If no hold is found, every field is `None`. `frames` is a per-frame table of raw and smoothed angles, angular velocities and stop flags.

**Settings** can be changed per call without editing the code, e.g.:

```python
frames, hold = process_video("clip.mp4", detector, velocity_threshold=20, gap_tolerance=0.3)
```

The defaults are a 15 °/s stop threshold, a 0.25 s gap tolerance, a 9-frame smoothing window of polynomial order 2, and a 45° limit for a plausible cross. Every parameter, with units, is described in the [API documentation](#documentation).

**Input requirements.**
- **Video:** any file OpenCV can read (`.mp4`, `.mov`), with one athlete in frame, the camera roughly front-on, and both arms and the hips visible.
- **Photos:** JPEG or PNG (convert HEIC first).

---

## Data

The raw videos and photos are **not** in this repository:

- **Personal recordings** (`HPCComp.mov`, `JoshCross1/2`, `IMG_2471.jpg`) are of the author, used with consent, and kept private.
- **The 2025 World Championship clips** (`2025Worlds1` to `7`) are short excerpts of broadcast footage. They are copyrighted and cannot be redistributed.

The notebook expects this layout in `DATA_DIR`:

```
<DATA_DIR>/
├── IMG_2471.jpg       front-on reference photo (sections 1 and 5)
├── HPCComp.mov        manually annotated validation clip (sections 2 and 3)
├── JoshCross1.mp4
├── JoshCross2.mp4
└── 2025Worlds1.mp4 … 2025Worlds7.mp4
```

**To run on your own clip:** put it in `DATA_DIR` and add its name to `CLIP_NAMES` in section 0, or call `process_video` directly as shown above.

**Running without the data:** all tests still run. They generate their own synthetic signals and a synthetic video.

**Safeguards:** `.gitignore` blocks `*.mp4`, `*.mov`, `*.MOV`, `*.HEIC` and `models/`, so footage and model files cannot be committed by accident.

---

## Tests

```bash
pytest                                                          # all tests
pytest -v --html=docs/test_report.html --self-contained-html    # with an HTML report
pytest --doctest-modules src                                    # check the examples in the docstrings
```

| File | Covers |
|---|---|
| `test_angle_calculation.py` | Angle maths, symmetry, deduction band edges, invalid input |
| `test_hold_detection.py` | Each processing step; synthetic holds with a known start, end and angle |
| `test_perspective.py` | Yaw estimate and angle correction on a synthetic rotated body |
| `test_pose_estimation.py` | Video loop on a generated video with a stand-in detector; skipped if MediaPipe or OpenCV is missing |
| `test_matches_original_notebook.py` | `src/` reproduces the original notebook implementation to 1e-12 |

The latest report is `docs/test_report.html`.

---

## Documentation

HTML API documentation is generated from the NumPy-style docstrings in `src/`:

```bash
pdoc src -o docs
```

- **View locally:** open `docs/index.html` in a browser.
- **View online:** https://joshua-walker978.github.io/iron-cross-judging/ (GitHub Pages, served from `docs/` on `main`).
- **Keeping it in sync:** regenerate after changing any docstring, and commit the docs together with the code.

---

## Development workflow

- **Branches:** larger changes are made on a branch and merged by pull request. For example, the move from a single notebook into the tested `src/` package was [pull request #1](https://github.com/Joshua-Walker978/iron-cross-judging/pull/1).
- **Commits:** one logical change per commit, with a message saying what changed, e.g. "Add per-clip run time to batch evaluation".
- **Before committing code:** run `pytest`, then `pdoc src -o docs`.
- **Committing from Colab:**
  - Code, docs and the test report go through section 7 of the notebook, which commits only when `COMMIT_MESSAGE` is set.
  - The notebook itself is saved with *File → Save a copy in GitHub*, at the path `notebooks/main_pipeline.ipynb`.
- **Design log:** findings and design decisions are recorded in `NOTES.md` as the project develops.
- **Early history:** commits before pull request #1 include repeated messages. Early versions of the notebook contained git commands that ran again on every *Run all*. The commit cell is now guarded so this cannot recur.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not open requirements file` in section 0 | Colab still holds an old clone: *Runtime → Disconnect and delete runtime*, then run again |
| `FileNotFoundError` when listing `DATA_DIR` | Check the path. The author's Drive folder name, `data raw `, ends with a space |
| `ModuleNotFoundError: No module named 'src'` | The working directory is not the repository root: `%cd` to it |
| `Could not read image` for a photo | OpenCV cannot read HEIC: convert to JPEG |
| `git push` asks for a password or is refused | Create a new personal access token with write access to this repository |
| Model download fails | The first run needs internet access to download the pose model |

---

## AI use

AI assistance (Claude, Anthropic) was used during development, including restructuring the code into `src/`, writing tests and docstrings, and drafting documentation. Details and a reflection are in the project report.

## Acknowledgements

- Pose estimation: [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (Google, Apache 2.0).
- Deduction rules: FIG Men's Artistic Gymnastics Code of Points 2025–2028.

## License

MIT; see [LICENSE](LICENSE).
