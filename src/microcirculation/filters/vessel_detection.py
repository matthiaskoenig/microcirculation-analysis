"""General filters for frame processing."""
import numpy as np
from PIL import Image
from pathlib import Path
import os
import cv2
from typing import Iterable

from microcirculation.filters.morphological_operations import *
from microcirculation.filters.standard_transformations import *
from microcirculation.utils import get_average_grayscale_value, get_image_segment
from microcirculation.filters.vessel_detection import *
from microcirculation import resources_dir, results_dir

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


def detect_vessels_in_frame(image: Image.Image, output_path: str, config: Iterable) -> Image.Image:
    if "global_hist" in config:
        image = histogram_equalization_global(image=image)
        output_path = output_path + "_ghe"
    if "local_hist" in config:
        image = histogram_equalization_local(image=image)
        output_path = output_path + "_lhe"
    if "gaussian" in config:
        radius = 1
        image = gaussian_filter(image=image, radius=radius)
        output_path = output_path + "_gn"
    if "canny" in config:
        image = canny_edge_detection(image=image)
        output_path = output_path + "_edges"
    if "ada_thresh" in config:
        image = adaptive_thresholding(image=image)
        output_path = output_path + "_adathresh"
    if "otsu" in config:
        image = otsu(image=image)
        output_path = output_path + "_otsu"
    if "median" in config:
        image = median_blur(image=image)
        output_path = output_path + "_md"

    return image, output_path


def vessel_detection_pipeline(image_path: Path, output_dir: Path, config: Iterable):
    output_path = output_dir / f"{image_path.stem}"

    frame = cv2.imread(str(image_path), 0)
    image = Image.fromarray(frame)

    image, output_path = detect_vessels_in_frame(image=image, output_path=str(output_path))
    
    output_path = str(output_path) + ".png"

    frame = np.array(image)
    cv2.imwrite(output_path, frame)


if __name__ == "__main__":
    image_path = results_dir / "frames" / "FMR_015-TP1-1_converted" / "FMR_015-TP1-1_converted_frame0.png"

    output_dir = results_dir / "pipeline_results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    configs = [
        ["global_hist", "local_hist", "ada_thresh", "median"]
    ]

    for config in configs:
        vessel_detection_pipeline(image_path=image_path, output_dir=output_dir, config=config)