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
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from vidstab import VidStab

from microcirculation.console import console


def stabilize_video(
    video_stabilize: Path,
    video_keypoints: Optional[Path],
    video_out: Path,
    smoothing_window: int,
):
    """Stabilize video based on keypoints in processed video."""
    console.print(f"stabilize_video: {video_stabilize}")
    start = time.time()

    if video_keypoints is None:
        # use the original video for keypoints if no video is provided
        video_keypoints = video_stabilize

    # FIXME: improve methods & visualize keypoints
    stabilizer = VidStab(kp_method="DENSE")

    stabilizer.gen_transforms(
        input_path=str(video_keypoints),
        smoothing_window=smoothing_window,
        show_progress=True,
    )

    stabilizer.apply_transforms(
        str(video_stabilize), str(video_out), output_fourcc="FFV1"
    )

    console.print(f"video stabilized in: {time.time() - start:.2f} seconds")

    # figures of stabilization trajectories
    fig1, (ax1, ax2) = stabilizer.plot_trajectory()
    fig1.savefig(
        video_out.parent / f"{video_out.stem}_trajectory.png", bbox_inches="tight"
    )

    fig2, (ax3, ax4) = stabilizer.plot_transforms()
    fig2.savefig(
        video_out.parent / f"{video_out.stem}_transforms.png", bbox_inches="tight"
    )
    plt.show()


def generate_keypoint_video(video_path: Path, kp_method: str = "SIFT") -> Path:
    """
    Calculate keypoints and visualize on video/frames
    example: https://www.oreilly.com/library/view/computer-vision-with/9781788472395/1ff16b52-a319-4c94-b02d-574c56c84f75.xhtml
    """
    # FIXME: update

    start_time = datetime.now()

    if "keypoint_videos" not in os.listdir(results_dir):
        os.mkdir(results_dir / "keypoint_videos")
    keypoint_video_path = (
        results_dir
        / "keypoint_videos"
        / f"{video_path.stem}_keypoints{video_path.suffix}"
    )

    video_frames, frame_size, frame_rate = extract_video_frames(video_path)

    video_out = cv2.VideoWriter(
        str(keypoint_video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        frame_rate,
        frame_size,
        False,
    )

    for frame in video_frames:
        keypoint_frame = draw_keypoints_on_frame(frame, kp_method)
        video_out.write(keypoint_frame)

    video_out.release()

    end_time = datetime.now()

    keypoint_detection_time = int((end_time - start_time).total_seconds())
    print(f"*** Keypoints detected in {stringify_time(keypoint_detection_time)} ***")

    return keypoint_video_path


if __name__ == "__main__":
    from microcirculation import data_dir

    video_stabilize = data_dir / "test" / "FMR_010-TP1-1_converted.avi"
    video_keypoints = data_dir / "test" / "FMR_010-TP1-1_vessels.avi"

    # framerate on the OPS videos is 30 frames/second
    smoothing_window = 60

    # stabilize_video(
    #     video_stabilize=video_stabilize,
    #     video_keypoints=None,
    #     video_out=data_dir / "test" / "FMR_010-TP1-1_raw_stable.avi",
    #     smoothing_window=smoothing_window,
    # )

    stabilize_video(
        video_stabilize=video_stabilize,
        video_keypoints=video_keypoints,
        video_out=data_dir / "test" / "FMR_010-TP1-1_vessels_stable.avi",
        smoothing_window=smoothing_window,
    )
