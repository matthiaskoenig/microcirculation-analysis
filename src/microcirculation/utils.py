from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def get_average_grayscale_value(image: Image.Image) -> int:
    image = np.array(image, dtype=np.uint8)
    return int(np.mean(image))


def extract_video_frames(video_in: Path) -> np.array:
    """Extract video frames."""
    video = cv2.VideoCapture(str(video_in))
    frame_height: int = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width: int = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_rate: int = int(video.get(cv2.CAP_PROP_FPS))

    frames = []
    while True:
        read_success, frame = video.read()

        if read_success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        else:
            break

    frame_size = (frame_width, frame_height)

    return frames, frame_size, frame_rate


def write_frames_as_video(
    frames: Iterable[np.array],
    frame_size: Iterable[int],
    frame_rate: float,
    video_out_path: Path,
) -> None:
    video_out_buffer = cv2.VideoWriter(
        str(video_out_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        frame_rate,
        frame_size,
        False,
    )

    for frame in frames:
        assert len(frame.shape) == 2
        frame = np.uint8(frame)
        video_out_buffer.write(frame)

    video_out_buffer.release()


def stack_images(images: Iterable[Image.Image]) -> Image.Image:
    """Stack given images."""
    widths, heights = zip(*(image.size for image in images))
    total_width = sum(widths)
    max_height = max(heights)

    stacked_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for image in images:
        stacked_image.paste(image, (x_offset, 0))
        x_offset += image.size[0]

    return stacked_image


def get_image_segment(
    image: Image,
    left_perc: int,
    right_perc: int,
    top_perc: int,
    bottom_perc: int,
) -> Image.Image:
    """
    Cropping out a segment of the image and returning the cropped segment.

    :param image: the image to be cropped
    :param left_perc: left border of the cropping region in terms of percentage of original image
    :param right_perc: right border of the cropping region in terms of percentage of original image
    :param top_perc: top border of the cropping region in terms of percentage of original image
    :param bottom_perc: bottom border of the cropping region in terms of percentage of original image
    """

    (width, height) = image.size
    region = (
        left_perc / 100 * width,
        top_perc / 100 * height,
        right_perc / 100 * width,
        bottom_perc / 100 * height,
    )
    return image.crop(region)
