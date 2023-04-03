"""Convert OPS videos in readable/playable format."""

import time
from pathlib import Path

import cv2
import numpy as np
from progress.bar import IncrementalBar

from microcirculation import data_dir
from microcirculation.console import console


def convert_video(
    input_path: Path,
    output_path: Path,
    fps_out: float,
):
    """Read video and convert to FFV1 codec with given frame rate."""
    console.print(f"Conversion: {input_path} -> {output_path}")
    start = time.time()
    cap = cv2.VideoCapture(str(input_path))

    # input video
    frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_rate: int = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)

    if not np.isclose(frame_rate, fps_out):
        console.print(f"changing framerate: {frame_rate} -> {fps_out}")

    # Define the codec and create VideoWriter object
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # Most codecs are lossy. If you want lossless video file you need to use a
    # lossless codecs (e.g. FFMPEG FFV1, Huffman HFYU, Lagarith LAGS, etc...)
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")  # lossless
    out = cv2.VideoWriter(
        filename=str(output_path),
        fourcc=fourcc,
        fps=fps_out,
        frameSize=frame_size,
    )
    bar = IncrementalBar("", max=frame_count, suffix="%(percent)d%%")
    while True:
        bar.next()
        read_success, frame = cap.read()
        if not read_success:
            break

        out.write(frame)

    cap.release()
    out.release()
    bar.finish()

    console.print(f"video converted in: {time.time() - start:.2f} seconds")


def convert_video_directory(input_dir, output_dir, fps_out: float):
    """Convert videos."""

    # TODO: parallelization
    # videos =

    for video_in in sorted(input_dir.glob("*.avi")):
        video_out = output_dir / f"{video_in.name}"
        convert_video(
            input_path=video_in,
            output_path=video_out,
            fps_out=fps_out,
        )


if __name__ == "__main__":
    # test videos
    # convert_video(
    #     input_path=data_dir / "test" / "rat_liver_ops_raw.avi",
    #     output_path=data_dir / "test" / "rat_liver_ops.avi",
    #     fps_out=30.0,  # slow down IDF images
    # )
    # convert_video(
    #     input_path=data_dir / "test" / "rat_liver_idf_raw.avi",
    #     output_path=data_dir / "test" / "rat_liver_idf.avi",
    #     fps_out=30.0,  # slow down IDF images
    # )
    # convert_video(
    #     input_path=data_dir / "test" / "human_sublingual_idf_raw.avi",
    #     output_path=data_dir / "test" / "human_sublingual.avi",
    #     fps_out=30.0,  # slow down IDF images
    # )

    # convert ops videos
    # convert_video_directory(
    #     input_dir=data_dir / "rat_liver_ops" / "videos" / "raw",
    #     output_dir=data_dir / "rat_liver_ops" / "videos" / "converted",
    #     fps_out=30.0,
    # )

    # convert_video_directory(
    #     input_dir=data_dir / "rat_liver_idf" / "videos" / "raw",
    #     output_dir=data_dir / "rat_liver_idf" / "videos" / "converted",
    #     fps_out=30.0,  # slow down idf videos
    # )

    # convert_video_directory(
    #     input_dir=data_dir / "human_sublingual_idf" / "videos" / "raw",
    #     output_dir=data_dir / "human_sublingual_idf" / "videos" / "converted",
    #     fps_out=30.0,  # slow down idf videos
    # )
