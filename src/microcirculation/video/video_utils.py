"""
1. Get information for given video (use cv library)
=>
"""
import copy
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2

composite_videos_path = Path("./composite_videos")


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
