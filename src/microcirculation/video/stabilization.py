"""
## Simple video stabilization using OpenCV
http://nghiaho.com/?p=2093

1. Find the transformation from previous to current frame using optical flow for all frames.
   The transformation only consists of three parameters: dx, dy, da (angle).
   Basically, a rigid Euclidean transform, no scaling, no sharing.
2. Accumulate the transformations to get the “trajectory” for x, y, angle, at each frame.
3. Smooth out the trajectory using a sliding average window.
   The user defines the window radius, where the radius is the number of frames used for smoothing.
4. Create a new transformation such that new_transformation = transformation + (smoothed_trajectory – trajectory).
5. Apply the new transformation to the video.

https://adamspannbauer.github.io/python_video_stab
pip install vidstab[cv2]

Necessary to find the correct keypoints:
https://www.pythonpool.com/opencv-keypoint/
how to deal with rotation?



"""
from pathlib import Path
import os
from datetime import datetime
from typing import Iterable

import matplotlib.pyplot as plt
from vidstab import VidStab

from microcirculation import results_path, resources_path
from microcirculation.utils import stringify_time
from microcirculation.video.video_utils import get_video_info, generate_vessel_detected_video

def stabilize_video(original_video_path: Path, vessels_video_path: Path):
    start_time = datetime.now()

    stabilizer = VidStab(kp_method="DENSE")

    smoothing_window: int = 35
    stabilizer.gen_transforms(
        input_path=str(vessels_video_path),
        smoothing_window=smoothing_window,  # FIXME: this must be adjusted
        show_progress=True,
    )

    if "stabilized_videos" not in os.listdir(results_path):
        os.mkdir(results_path / "stabilized_videos")
    if original_video_path.stem not in os.listdir(results_path / "stabilized_videos"):
        os.mkdir(results_path / "stabilized_videos" / original_video_path.stem)

    stabilized_video_path = results_path / "stabilized_videos" / original_video_path.stem / f"{original_video_path.stem}_stabilized{original_video_path.suffix}"
    stabilizer.apply_transforms(str(original_video_path), str(stabilized_video_path))

    fig1, (ax1, ax2) = stabilizer.plot_trajectory()
    fig1.savefig(
        stabilized_video_path.parent / f"trajectory.png", bbox_inches="tight"
    )

    fig2, (ax3, ax4) = stabilizer.plot_transforms()
    fig2.savefig(
        stabilized_video_path.parent / f"transforms.png", bbox_inches="tight"
    )

    end_time = datetime.now()

    stabilization_time = int((end_time - start_time).total_seconds())
    print(f"Video stabilized in {stringify_time(stabilization_time)}")


if __name__ == "__main__":

    video_path = resources_path / "BRM-TC-Jena-P0-AdHoc-1-20220901-092449047---V0.avi"
    detection_config = ["global_hist", "ada_thresh", "median"]
    stabilize_video(video_path=video_path, detection_config=detection_config)
