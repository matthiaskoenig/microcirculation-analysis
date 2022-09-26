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


from vidstab import VidStab
import matplotlib.pyplot as plt

from microcirculation import data_path


def stabilize_video(video_in: Path, video_out: Path):

    stabilizer = VidStab(
        # processing_max_dim=300
        kp_method="DENSE"
    )
    print(video_in)

    # 1. stabilize the heart rate (window of ~ 1second)
    fps: float = 30.0
    stabilizer.stabilize(
        input_path=str(video_in),
        output_path=str(video_out),
        smoothing_window=2,  # int(fps),  # FIXME: this must be adjusted
        output_fourcc="FFV1",
        show_progress=True,
        playback=False,
    )

    fig1, (ax1, ax2) = stabilizer.plot_trajectory()

    fig1.savefig(video_in.parent / f"{video_out.stem}_trajectory.png", bbox_inches="tight")
    plt.show()

    fig2, (ax3, ax4) = stabilizer.plot_transforms()
    fig2.savefig(video_in.parent / f"{video_out.stem}_transforms.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # convert ops videos
    # input_dir: Path = data_path / "ops"
    # for video_in in sorted(input_dir.glob('*.avi')):
    #     if "converted" in str(video_in):
    #         print(video_in)
    #         video_stable = video_in.parent / f"{video_in.stem.replace('_converted')}_stable.avi"
    #         stabilize_video(video_in=video_in, video_out=video_stable)

    video_in: Path = data_path / "ops" / "FMR_015-TP1-2_converted.avi"
    video_stable: Path = data_path / "ops" / "FMR_015-TP1-2_stable2.avi"
    stabilize_video(video_in=video_in, video_out=video_stable)
    # video_in: Path = data_path / "braedius" / "output.avi"
    # video_stable: Path = data_path / "braedius" / "output_stable.avi"
