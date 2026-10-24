"""
Automated judging of the gymnastics iron cross from video.

This package measures how far a gymnast's arms deviate from horizontal during
an iron cross on still rings, finds the held part of the element, and
classifies both the arm angle and the hold duration into the deduction
categories of the FIG Men's Artistic Gymnastics Code of Points (2025-2028).

Pipeline
--------
1. **Pose estimation** (`src.pose_estimation`): MediaPipe Pose Landmarker
   returns 33 body landmarks per video frame.
2. **Arm angle** (`src.angle_calculation`): the shoulder-to-wrist vector of
   each arm is converted to a deviation from horizontal in degrees.
3. **Hold detection** (`src.hold_detection`): the per-frame deviation signal
   is smoothed, differentiated to an angular velocity, thresholded into
   "stopped" frames, merged into segments, and the longest plausible
   segment is taken as the hold.
4. **Classification**: the hold's mean angle and its duration are mapped to
   deduction categories.
5. **Perspective correction** (`src.perspective`, experimental): an attempt
   to correct the measured angle for camera yaw using body proportions.

Dependencies
------------
Only `src.pose_estimation` needs MediaPipe and OpenCV. The other modules use
only NumPy, pandas and SciPy, so they can be unit tested without a video or
a pose model.

Units
-----
Angles are in degrees, times in seconds, angular velocities in degrees per
second, and image coordinates in pixels with the origin at the top-left and
y increasing downwards.
"""

__docformat__ = "numpy"
