from typing import Iterable

import numpy as np
from PIL import Image


def get_average_grayscale_value(image: Image.Image) -> int:
    image = np.array(image, dtype=np.uint8)
    return int(np.mean(image))


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
