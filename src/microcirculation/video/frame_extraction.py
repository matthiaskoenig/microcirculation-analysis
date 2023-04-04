import os
import shutil
from pathlib import Path

import cv2

from microcirculation import resources_dir, results_dir
from microcirculation.video.video_utils import get_video_info


def extract_frames_from_video(
    video_path: Path, frame_capture_interval: int, results_dir: Path
) -> None:
    """
    Extracts frames from a video at interval of some frames and saves them as images

    @param video_path: input video path
    @param frame_capture_interval: capture a frame once for these many frames
    @param results_dir: output directory where extracted frames will be saved
    """

    video_name = video_path.stem

    output_dir = results_dir / video_name
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    video = cv2.VideoCapture(str(video_path))

    i = 0
    while video.isOpened():
        read_success, frame = video.read()
        if read_success:
            if i % frame_capture_interval == 0:
                # capture this frame
                output_frame_path = output_dir / f"{video_name}_frame{i}.png"
                cv2.imwrite(str(output_frame_path), frame)
        else:
            break

        i += 1

    video.release()


if __name__ == "__main__":
    video_names = [
        "BRM-TC-Jena-P0-AdHoc-1-20220901-092449047---V0.avi",
        "FMR_015-TP1-1_converted.avi",
    ]

    FRAMES_NEEDED_PER_VIDEO = 10

    for video_name in video_names:
        video_path = resources_dir / video_name

        frames_results_dir = results_dir / "frames"
        if "frames" not in os.listdir(results_dir):
            os.mkdir(frames_results_dir)

        frame_count = get_video_info(video_path=video_path)["frame_count"]
        frame_capture_interval = frame_count // FRAMES_NEEDED_PER_VIDEO

        extract_frames_from_video(
            video_path=video_path,
            frame_capture_interval=frame_capture_interval,
            results_dir=frames_results_dir,
        )

        print(f"*** {FRAMES_NEEDED_PER_VIDEO} Frames extracted for {video_name} ***")
