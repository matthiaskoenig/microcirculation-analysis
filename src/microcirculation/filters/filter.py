"""General filters for frame processing."""
import numpy as np
from PIL import Image

from microcirculation.filters.morphological_operations import *
from microcirculation.filters.standard_transformations import *
from microcirculation.utils import get_average_grayscale_value, get_image_segment

__all__ = [
    "threshold_vessels_detection",
    "threshold_vessels_detection_local",
    "threshold_vessels_detection_avg_grayscale",
    "morphological_vessels_detection",
    "morpho_closing_vessels_detection",
    "blur_erosion_vessels_detection",
]


def threshold_vessels_detection(image: Image.Image, value: int = 130) -> Image.Image:
    """Detection of vessels by thresholding edge detected image.

    :param value: threshold value
    """
    # image = detect_edges_sobel(image=image)

    image_0 = get_image_segment(
        image=image, left_perc=0, right_perc=15, top_perc=0, bottom_perc=100
    )
    image_0 = gamma_transform(image_0, gamma=-0.25)
    image_0 = threshold(image=image_0, value=120, invert=True)

    image_1 = get_image_segment(
        image=image, left_perc=15, right_perc=35, top_perc=0, bottom_perc=100
    )
    image_1 = gamma_transform(image_1, gamma=-0.3)
    image_1 = threshold(image=image_1, value=100, invert=True)

    image_11 = get_image_segment(
        image=image, left_perc=35, right_perc=45, top_perc=0, bottom_perc=100
    )
    image_11 = gamma_transform(image_11, gamma=-0.2)
    image_11 = threshold(image=image_11, value=45, invert=True)

    image_2 = get_image_segment(
        image=image, left_perc=80, right_perc=100, top_perc=0, bottom_perc=100
    )
    image_2 = gamma_transform(image_2, gamma=-0.2)
    image_2 = threshold(image=image_2, value=40, invert=True)

    image = threshold(image=image, value=140)

    image.paste(image_0, (0, 0))
    image.paste(image_1, (int(0.15 * image.size[0]), 0))
    image.paste(image_11, (int(0.35 * image.size[0]), 0))
    image.paste(image_2, (int(0.80 * image.size[0]), 0))

    return image


def threshold_vessels_detection_local(
    image: Image.Image, value: int = 130
) -> Image.Image:
    """Detection of vessels by thresholding edge detected image.

    :param value: threshold value
    """
    image = histogram_equalization_local(image=image)

    # image_0 = get_image_segment(
    #     image=image, left_perc=0, right_perc=35, top_perc=0, bottom_perc=100
    # )
    # image_0 = gamma_transform(image_0, gamma=-0.2)
    # image_0 = threshold(image=image_0, value=65, invert=True)

    image = threshold(image=image, value=100)

    # image.paste(image_0, (0, 0))

    # image = median_blur(image=image)

    return image


def threshold_vessels_detection_avg_grayscale(image: Image.Image) -> Image.Image:
    """
    Detection of vessels by thresholding edge detected image using avgerage grayscale value.
    """
    image_edges = detect_edges_sobel(image)

    avg_grayscale_value: int = get_average_grayscale_value(image_edges)
    return threshold(image_edges, avg_grayscale_value)


def morphological_vessels_detection(image: Image.Image) -> Image.Image:
    """
    Isolation of vessels by applying multiple morphological operations on the image
    """

    image = threshold(image, 130)
    image = closing(image)
    image = dilate(image)
    image = erode(image, (5, 5))
    return image


def morpho_closing_vessels_detection(image: Image.Image) -> Image.Image:
    """
    Isolation of vessels by applying closing operation on thresholded image
    """
    # FIXME: what is this threshold
    image1 = threshold(image, 130)
    return closing(image1)


def blur_erosion_vessels_detection(image: Image.Image) -> Image.Image:
    """
    Isolation of vessels by applying erosion operation on gaussian blurred image
    """
    image_blur: Image.Image = gaussian_filter(image)
    image = Image.fromarray(np.array(image) - np.array(image_blur))
    return erode(image)
