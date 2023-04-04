"""Convert OPS videos in readable/playable format."""

import time
from pathlib import Path
from typing import Iterable, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm

from microcirculation import data_dir

import ray
ray.init()


@ray.remote
def convert_video(
    video_in: Path,
    video_out: Path,
    fps_out: float,
):
    """Read video and convert to FFV1 codec with given frame rate."""
    print(f"Conversion: {video_in} -> {video_out}")
    start = time.time()
    cap = cv2.VideoCapture(str(video_in))

    # input video
    frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_rate: int = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)

    if not np.isclose(frame_rate, fps_out):
        print(f"changing framerate: {frame_rate} -> {fps_out}")

    # Define the codec and create VideoWriter object
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # Most codecs are lossy. If you want lossless video file you need to use a
    # lossless codecs (e.g. FFMPEG FFV1, Huffman HFYU, Lagarith LAGS, etc...)
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")  # lossless
    out = cv2.VideoWriter(
        filename=str(video_out),
        fourcc=fourcc,
        fps=fps_out,
        frameSize=frame_size,
    )
    while True:
        read_success, frame = cap.read()
        if not read_success:
            break

        out.write(frame)

    cap.release()
    out.release()

    print(f"{video_in.name} converted: {time.time() - start:.2f} seconds")


def convert_video_directory(input_dir: Path, output_dir: Path, fps_out: float):
    """Convert videos."""
    if not input_dir.exists():
        raise IOError(f"input_dir does not exist: {input_dir}")

    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    data = []
    for video_in in sorted(input_dir.glob("*.avi")):
        data.append(
            {
                'video_in': video_in,
                'video_out': output_dir / f"{video_in.name}",
                'fps_out': fps_out,
            }
        )
    convert_videos(data)

def to_iterator(obj_ids):
    """Iterator for the progress bar."""
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

def convert_videos(data: Iterable[Dict[str, Any]]):
    """Process videos in parallel with ray."""
    obj_ids = [convert_video.remote(**d) for d in data]
    # results = [ray.get(obj_id) for obj_id in obj_ids]
    #
    for _ in tqdm(to_iterator(obj_ids), total=len(obj_ids)):
        pass


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

    convert_video_directory(
        input_dir=data_dir / "human_sublingual_idf" / "videos" / "raw",
        output_dir=data_dir / "human_sublingual_idf" / "videos" / "converted",
        fps_out=30.0,  # slow down idf videos
    )
