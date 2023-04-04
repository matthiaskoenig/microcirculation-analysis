"""Processing of videos by applying filters to single frames."""

import time
from pathlib import Path
from typing import Iterable, Dict, Any
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

from microcirculation.filters.vessel_detection import detect_vessels_in_frame
import ray
ray.init()


def rayids_to_iterator(obj_ids):
    """Iterator for the progress bar."""
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

@ray.remote
def process_video(
    video_in: Path, video_out: Path, filters: Iterable[str],
) -> None:
    """Process video for stabilization.

    Apply filters and preprocessing which will hopefully improve image stabilization.
    Detects vessel features for better calculation of key points in the stabilization.
    """
    print(f"process_video: {video_in.name} -> {video_out.name}, {filters}")
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
                config=filters,
            )
            writer.write(np.array(vessel_frame))
        else:
            break

    capture.release()
    writer.release()

    print(f"{video_in.name} processed: {time.time() - start:.2f} seconds")


def process_video_directory(input_dir: Path, output_dir: Path, filters: Iterable[str]):
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
                'filters': filters,
            }
        )
    process_videos(data)

def to_iterator(obj_ids):
    """Iterator for the progress bar."""
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

def process_videos(data: Iterable[Dict[str, Any]]):
    """Process videos in parallel with ray."""
    obj_ids = [process_video.remote(**d) for d in data]
    for _ in tqdm(rayids_to_iterator(obj_ids), total=len(obj_ids)):
        pass


if __name__ == "__main__":
    from microcirculation import data_dir

    # videos = [
    #     {
    #         'video_in': data_dir / "test" / "rat_liver_idf.avi",
    #         'video_out': data_dir / "test" / "rat_liver_idf_lhe.avi",
    #         'filters': ["local_hist"],
    #     },
    #     {
    #         'video_in': data_dir / "test" / "rat_liver_ops.avi",
    #         'video_out': data_dir / "test" / "rat_liver_ops_lhe.avi",
    #         'filters': ["local_hist"],
    #     },
    #     {
    #         'video_in': data_dir / "test" / "human_sublingual_idf.avi",
    #         'video_out': data_dir / "test" / "human_sublingual_lhe.avi",
    #         'filters': ["local_hist"],
    #     },
    #     {
    #         'video_in': data_dir / "test" / "rat_liver_idf.avi",
    #         'video_out': data_dir / "test" / "rat_liver_idf_lhe.avi",
    #         'filters': ["local_hist"],
    #     },
    #     {
    #         'video_in': data_dir / "test" / "rat_liver_ops.avi",
    #         'video_out': data_dir / "test" / "rat_liver_ops_lhe.avi",
    #         'filters': ["local_hist"],
    #     },
    #     {
    #         'video_in': data_dir / "test" / "human_sublingual_idf.avi",
    #         'video_out': data_dir / "test" / "human_sublingual_lhe.avi",
    #         'filters': ["local_hist"],
    #     }
    # ]

    # process_video_directory(
    #     input_dir=data_dir / "rat_liver_ops" / "videos" / "converted",
    #     output_dir=data_dir / "rat_liver_ops" / "videos" / "processed_lhe",
    #     filters=["local_hist"]
    # )

    # process_video_directory(
    #     input_dir=data_dir / "rat_liver_idf" / "videos" / "converted",
    #     output_dir=data_dir / "rat_liver_idf" / "videos" / "processed_lhe",
    #     filters=["local_hist"]
    # )
    # process_video_directory(
    #     input_dir=data_dir / "human_sublingual_idf" / "videos" / "converted",
    #     output_dir=data_dir / "human_sublingual_idf" / "videos" / "processed_lhe",
    #     filters=["local_hist"]
    # )

    ray.get(process_video.remote(
        video_in=data_dir / "test" / "rat_liver_idf.avi",
        video_out=data_dir / "test" / "rat_liver_idf_lhe_at_median.avi",
        filters=["local_hist", "ada_thresh", "median"],
    ))
