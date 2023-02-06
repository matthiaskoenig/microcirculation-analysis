"""General filters for frame processing."""
from PIL import Image
import numpy as np
from pathlib import Path
import cv2

from microcirculation import resources_path, results_path
from microcirculation.filters.custom_detector.standard_transformations import *
from microcirculation.filters.custom_detector.morphological_operations import *

from microcirculation.utils import get_average_grayscale_value, stack_images, get_image_segment, keypoint_detection

__all__ = [
    "threshold_vessels_detection",
    "threshold_vessels_detection_local",
]


def threshold_vessels_detection(image: Image.Image, value: int = 130) -> Image.Image:
    """Detection of vessels by thresholding edge detected image.

    :param value: threshold value
    """
    # image = detect_edges_sobel(image=image)

    image_0 = get_image_segment(image=image, left_perc=0, right_perc=15, top_perc=0, bottom_perc=100)
    image_0 = gamma_transform(image_0, gamma=-0.25)
    image_0= threshold(image=image_0, value=120, invert=True)
    
    image_1 = get_image_segment(image=image, left_perc=15, right_perc=35, top_perc=0, bottom_perc=100)
    image_1 = gamma_transform(image_1, gamma=-0.3)
    image_1 = threshold(image=image_1, value=100, invert=True)

    image_11 = get_image_segment(image=image, left_perc=35, right_perc=45, top_perc=0, bottom_perc=100)
    image_11 = gamma_transform(image_11, gamma=-0.2)
    image_11 = threshold(image=image_11, value=45, invert=True)

    image_2 = get_image_segment(image=image, left_perc=80, right_perc=100, top_perc=0, bottom_perc=100)
    image_2 = gamma_transform(image_2, gamma=-0.2)
    image_2 = threshold(image=image_2, value=40, invert=True)

    image = threshold(image=image, value=140)
    
    image.paste(image_0, (0, 0))
    image.paste(image_1, (int(0.15*image.size[0]), 0))
    image.paste(image_11, (int(0.35*image.size[0]), 0))
    image.paste(image_2, (int(0.80*image.size[0]), 0))

    return image

def threshold_vessels_detection_local(image: Image.Image, value: int = 130) -> Image.Image:
    """Detection of vessels by thresholding edge detected image.

    :param value: threshold value
    """

    image = local_histogram_equalization(image=image)

    image_0 = get_image_segment(image=image, left_perc=0, right_perc=35, top_perc=0, bottom_perc=100)
    image_0 = gamma_transform(image_0, gamma=-0.2)
    image_0 = threshold(image=image_0, value=65, invert=True)

    image = threshold(image=image, value=150)

    image.paste(image_0, (0, 0))

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


def apply_all_filters(image_path: Path, results_dir: Path) -> None:
    """Apply all filter pipelines to given image."""

    results_dir.mkdir(exist_ok=True, parents=True)

    for k, f_filter_pipeline in enumerate([
        threshold_vessels_detection,
        threshold_vessels_detection_local,
        threshold_vessels_detection_avg_grayscale,
        morphological_vessels_detection,
        morpho_closing_vessels_detection,
        blur_erosion_vessels_detection,
    ]):
        # read the image
        image_original: Image.Image = Image.open(image_path)
        # convert to greyscale
        image_grey = image_original.convert("L")

        if k == 0:
            image_original.save(str(results_dir / f"00_{test_image_path.name}"))
            image_grey.save(str(results_dir / f"00_{test_image_path.stem}_grey.png"))

        # apply filter
        print(f"*** Apply {f_filter_pipeline.__name__} ***")
        image_filtered = f_filter_pipeline(image_grey)
        print(image_filtered)
        # stack images
        # image_out: Image.Image = stack_images([image_original, image_filtered])
        image_out: Image.Image = stack_images([image_filtered])
        # image_out = image_filtered
        # save image
        image_out_path = results_dir / f"0{k+1}_{test_image_path.stem}_{f_filter_pipeline.__name__}.png"

        image_out.save(str(image_out_path))


def superimpose_keypoints_on_image(image_path: Path, results_dir: Path) -> None:

    results_dir.mkdir(exist_ok=True, parents=True)

    image: Image.Image = Image.open(image_path)
    keypoints_image: Image.Image = Image.fromarray(keypoint_detection(image, "SIFT"))
    
    keypoints_image_path = results_dir / f"{image_path.stem}_keypoints.png"
    keypoints_image.save(str(keypoints_image_path))


if __name__ == "__main__":
    results_dir: Path = results_path / "filter_pipelines"
    test_image_path: Path = resources_path / "sublingua.png"
    apply_all_filters(image_path=test_image_path, results_dir=results_dir)
    
    keypoints_results_dir: Path = results_dir / "keypoints"
    image_for_keypoints: Path = results_dir / "01_sublingua_threshold_vessels_detection.png"
    superimpose_keypoints_on_image(image_path=image_for_keypoints, results_dir=keypoints_results_dir)

