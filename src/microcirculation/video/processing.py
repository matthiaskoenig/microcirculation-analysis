"""Processing of videos by applying filters to single frames."""

import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from microcirculation.console import console

from microcirculation.filters.vessel_detection import detect_vessels_in_frame
from progress.bar import IncrementalBar



def process_video(
    video_in: Path, video_out: Path, filters: Iterable[str], **kwargs
) -> None:
    """Process video for stabilization.

    Apply filters and preprocessing which will hopefully improve image stabilization.
    Detects vessel features for better calculation of key points in the stabilization.
    """
    console.print(f"process_video: {video_in} -> {video_out}")
    console.print(f"detection_config: {filters}")
    start = time.time()

    # input video
    capture = cv2.VideoCapture(str(video_in))
    frame_count: int = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
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

    bar = IncrementalBar('', max=frame_count, suffix='%(percent)d%%')
    while True:
        bar.next()
        read_success, frame = capture.read()
        if read_success:
            # FIXME: avoid unnessesary conversions
            # convert to greyscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(frame)
            vessel_frame, _ = detect_vessels_in_frame(
                image=image,
                config=filters,
            )
            writer.write(np.array(vessel_frame))
        else:
            break

    capture.release()
    writer.release()
    bar.finish()

    console.print(f"video processed in: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    from microcirculation import data_dir
    video_in = data_dir / "test" / "FMR_010-TP1-1_converted.avi"

    for video_out, filters in [
        (
            data_dir / "test" / "FMR_010-TP1-1_filters1.avi",
            ["local_hist"]
        ),
        # (
        #     data_dir / "test" / "FMR_010-TP1-1_vessels.avi",
        #     ["global_hist", "ada_thresh", "median"]
        # ),
    ]:
        process_video(
            video_in=video_in,
            video_out=video_out,
            filters= filters,
        )

