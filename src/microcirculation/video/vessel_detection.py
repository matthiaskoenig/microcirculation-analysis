import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from microcirculation.console import console

from microcirculation.filters.vessel_detection import detect_vessels_in_frame


def process_video(
    video_in: Path, video_out: Path, detection_config: Iterable[str]
) -> None:
    """Process video for stabilization.

    Apply filters and preprocessing which will hopefully improve image stabilization.
    Detects vessel features for better calculation of key points in the stabilization.
    """
    console.print(f"process_video: {video_in} -> {video_out}")
    console.print(f"detection_config: {detection_config}")
    start = time.time()

    # input video
    capture = cv2.VideoCapture(str(video_in))
    frame_height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_rate: int = int(capture.get(cv2.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)

    # output video
    # lossless codecs (e.g. FFMPEG FFV1, Huffman HFYU, Lagarith LAGS, etc...)
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"FFV1"),  # lossless coded
        frame_rate,
        frame_size,
        False,
    )

    while True:
        read_success, frame = capture.read()
        if read_success:
            # FIXME: avoid unnessesary conversions
            # convert to greyscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(frame)
            vessel_frame, _ = detect_vessels_in_frame(
                image=image,
                config=detection_config,
            )
            writer.write(np.array(vessel_frame))
        else:
            break

    capture.release()
    writer.release()

    console.print(f"video processing: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    from microcirculation import data_dir
    video_in = data_dir / "test" / "FMR_010-TP1-1_converted.avi"
    video_out = data_dir / "test" / "FMR_010-TP1-1_vessels.avi"
    process_video(
        video_in=video_in,
        video_out=video_out,
        detection_config=["global_hist", "ada_thresh", "median"]
    )
