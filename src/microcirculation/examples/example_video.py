"""Code for video normalization.

Defines a list of preprocess functions which allows to simplify the normalization.
"""
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image

from microcirculation.filters.filter import *
from microcirculation.utils import extract_video_frames, write_frames_as_video


def apply_preprocess_filters(
    video_in: Path, video_out_base: Path, filters=Iterable[Callable]
) -> Iterable[Path]:
    """Apply preprocess filters for better video normalization."""
    video_out_paths = []
    
    # apply filters (for vessel detection)
    for filter in filters:
        video = cv2.VideoCapture(str(video_in))
        video_out = f"{video_out_base.stem}_{filter.__name__}{video_out_base.suffix}"
        print(video_out)
        video_out_paths.append(Path(video_out))
        video_out_buffer = cv2.VideoWriter(
            video_out,
            cv2.VideoWriter_fourcc(*'MJPG'),
            int(video.get(cv2.CAP_PROP_FPS)),
            (int(video.get(3)), int(video.get(4))),
            False
        )

        while True:
            read_success, frame = video.read()

            if read_success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                image = Image.fromarray(frame)
                image = filter(image)
                # why conversion to uint8?
                frame = np.uint8(np.array(image))
                
                video_out_buffer.write(frame)
            else:
                break
    
        video_out_buffer.release()
    return video_out_paths


def apply_brightness_normalization(video_paths: Iterable[Path]) -> None:
    for video_path in filtered_video_paths:
        frames, frame_size, frame_rate = extract_video_frames(video_in=video_path)
        normalized_frames = normalize_frames_brightness(frames=frames)
        print(type(normalized_frames))
        write_frames_as_video(
            frames = normalized_frames, 
            frame_size = frame_size,
            frame_rate = 20.0,
            video_out_path = Path(f"{video_path.stem}_normalized{video_path.suffix}")
        )


if __name__ == "__main__":
    from microcirculation.filters.standard_transformations import (
        normalize_frames_brightness,
    )

    # input video
    video_path = (
        Path("/Users/maniklaldas/Desktop/BRM-TC-Jena-P0-AdHoc-1-20220901-092449047---V0.avi")
    )

    # detect vessels using different filters
    filtered_video_paths = apply_preprocess_filters(
        video_in=video_path,
        video_out_base=video_path.parent / f"{video_path.stem}_vessels{video_path.suffix}",
        filters=[
            threshold_vessels_detection,
            threshold_vessels_detection_local,
            threshold_vessels_detection_avg_grayscale,
            morphological_vessels_detection,
            morpho_closing_vessels_detection,
            blur_erosion_vessels_detection,
        ]
    )

    apply_brightness_normalization(video_paths=filtered_video_paths)
