"""Code for video normalization.

Defines a list of preprocess functions which allows to simplify the normalization.
"""
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image


def apply_preproces_filters(
    video_in: Path, video_out: Path, filters=Iterable[Callable]
) -> None:
    """Apply preprocess filters for better video normalization."""
    video = cv2.VideoCapture(str(video_in))

    video_out = cv2.VideoWriter(
        video_out,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        int(video.get(cv2.CAP_PROP_FPS)),
        (int(video.get(3)), int(video.get(4))),
    )

    while True:
        read_success, frame = video.read()

        if read_success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # apply filters
            for filter in filters:
                image = Image.fromarray(frame)
                image = filter(image)
                # why conversion to uint8?
                # frame = np.uint8(frame)

                video_out.write(np.array(image))

        else:
            break


if __name__ == "__main__":
    from microcirculation.filters.standard_transformations import (
        normalize_frames_brightness,
    )

    # input video
    video_path = (
        "/Users/maniklaldas/Desktop/BRM-TC-Jena-P0-AdHoc-1-20220901-092449047---V0.avi"
    )

    # detect vessels
    apply_preproces_filters(
        video_in=video_path,
        video_out=video_path.parent / f"{video_path.stem}_vessels{video_path.suffix}",
        filters=[],
    )

    # vessel_detected_video_path = preprocess_detect_vessel(video_in_path=video_path)

    normalized_frames = normalize_frames_brightness(frames=frames)
    # print("Vessels detected")
    # normalized_brightness_video_path = normalize_video_brightness(video_path=vessel_detected_video_path)
    # print(normalized_brightness_video_path)

    # detect_vessels_and_display(video_path=video_path)
