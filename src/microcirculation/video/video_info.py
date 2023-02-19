"""
1. Get information for given video (use cv library)
=>
"""
import copy
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

composite_videos_path = Path("./composite_videos")


def get_video_info(video_path: Path):
    """
    1. Get video meta data
        frame_height: int
        frame_width: int
        frame_rate: int  # frames per second
        pixel_width: float  # [µm] (magnification, ... spacial resolution; -> dictionary with defintions; OPS_10x =; OPS_5x ... IDF = ...)
        pixel_height: float  # [µm] not always square
    """

    video_file = cv2.VideoCapture(str(video_path))
    frame_count: int = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height: int = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width: int = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_rate: int = int(video_file.get(cv2.CAP_PROP_FPS))
    duration: int = frame_count // frame_rate  # duration in seconds

    pixel_width = 0  # TODO: need to figure out
    pixel_height = 0  # TODO: need to figure out

    return {
        "frame_count": frame_count,
        "frame_rate": frame_rate,
        "duration": duration,
        "frame_height": frame_height,
        "frame_width": frame_width,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
    }


def get_composite_video(
    video_1_path: Path, video_2_path: Path, alignment: str = "horizontal"
):
    """
    2.
    Create composite video: unstabilized/stabilized;
    => make video double the width: with left
    => clue 2 videos:
    call ffmpeg from python:
    => create example for: IDF sublingual, IDF rat, OPS rat (status quo)
    """

    assert alignment == "vertical" or alignment == "horizontal"

    video_1_info = get_video_info(video_1_path)
    print(video_1_info)
    video_2_info = get_video_info(video_2_path)
    print(video_2_info)

    if "composite_videos" not in os.listdir("../"):
        os.mkdir("./composite_videos")

    composite_file_name = (
        alignment
        + "_"
        + str(video_1_path).replace("/", "_")
        + "_"
        + str(video_2_path).replace("/", "_")
    )
    composite_file_path = composite_videos_path / composite_file_name

    if alignment == "horizontal":
        assert video_1_info["frame_height"] == video_2_info["frame_height"]
        subprocess.call(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_1_path),
                "-i",
                str(video_2_path),
                "-filter_complex",
                "hstack=inputs=2",
                str(composite_file_path),
            ]
        )
    else:
        assert video_1_info["frame_width"] == video_2_info["frame_width"]
        subprocess.call(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_1_path),
                "-i",
                str(video_2_path),
                "-filter_complex",
                "vstack=inputs=2",
                str(composite_file_path),
            ]
        )

    video_file = cv2.VideoCapture(str(composite_file_path))
    while video_file.isOpened():
        ret, frame = video_file.read()
        if ret:
            cv2.imshow(composite_file_name, frame)
            cv2.setWindowProperty(composite_file_name, cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            break

    video_file.release()
    cv2.destroyAllWindows()


# video_path = Path("/Users/maniklaldas/Desktop/BRM-TC-Jena-P0-AdHoc-1-20220901-092449047---V0.avi")
video_path = Path("/Users/maniklaldas/Desktop/FMR_015-TP1-1_converted.avi")

# keypoint_video_path = Path(get_keypoints_and_display(video_path, "SIFT"))

# get_composite_video(video_path, keypoint_video_path, "horizontal")


"""
3. Overview table of the videos
=> excel spreadsheet: especially important for the OPS videos; what degree of hepatectomy
=> Matthias: mapping of video ids to biological information; 
species: human/rat
tissue: liver/sublingual
"""
