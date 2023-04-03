"""Convert OPS videos in readable/playable format."""

from pathlib import Path
from typing import Tuple

import cv2 as cv

from microcirculation import data_dir
from microcirculation.console import console


def ops_conversion(
    input_path: Path,
    output_path: Path,
    fps_out: float,
    frame_size: Tuple[int, int],
):
    """Read OPS AVI and convert to correct avi."""
    console.print(f"Conversion: {input_path} -> {output_path}")
    cap = cv.VideoCapture(str(input_path))

    # Define the codec and create VideoWriter object
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # Most codecs are lossy. If you want lossless video file you need to use a
    # lossless codecs (e.g. FFMPEG FFV1, Huffman HFYU, Lagarith LAGS, etc...)
    fourcc = cv.VideoWriter_fourcc(*"FFV1")  # lossless
    out = cv.VideoWriter(
        filename=str(output_path),
        fourcc=fourcc,
        fps=fps_out,
        frameSize=frame_size,
    )
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            # print("Can't receive frame (stream end?). Exiting ...")
            break

        out.write(frame)

    cap.release()
    out.release()


def convert_ops_videos():
    """Convert all OPS videos."""
    input_dir: Path = data_dir / "rat_liver_ops" / "videos" / "raw"
    output_dir: Path = data_dir / "rat_liver_ops" / "videos" / "processed"

    for video_in in sorted(input_dir.glob("*.avi")):
        video_out = output_dir / f"{video_in.name}"
        ops_conversion(
            input_path=video_in,
            output_path=video_out,
            fps_out=30.0,
            frame_size=(640, 480),
        )


if __name__ == "__main__":
    # convert single video
    # input_path: Path = data_dir / "test" / "FMR_010-TP1-1.avi"
    # output_path: Path = data_dir / "test" / "FMR_010-TP1-1_converted.avi"
    # ops_conversion(
    #     input_path=input_path,
    #     output_path=output_path,
    #     fps_out=30.0,
    #     frame_size=(640, 480),
    # )

    # convert all videos
    convert_ops_videos()
