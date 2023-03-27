import numpy as np
from PIL import Image
from pathlib import Path
import os
import cv2
from typing import Iterable

from microcirculation.filters.morphological_operations import *
from microcirculation.filters.standard_transformations import *
from microcirculation.utils import get_average_grayscale_value, get_image_segment
from microcirculation.filters.filter import *
from microcirculation import resources_path, results_path

def visualize_green_channel(image_path: Path) -> None:
    image = cv2.imread(str(image_path))
    result = Image.fromarray(image[:, :, 1])    # B G R image

    result = histogram_equalization_local(image=result)
    # result = threshold_vessels_detection_local(image=result, value=120)
    # result = median_blur(image=result, size=3)
    result = gamma_transform(image=result, gamma=0.5)

    result = perform_grid_thresholding(image=result, size=250)

    result_path = f"{image_path.parent}/{image_path.stem}_green{image_path.suffix}"    
    cv2.imwrite(result_path, np.array(result))


def perform_grid_thresholding(image: Image, size: int) -> Image.Image:
    frame = np.array(image)
    (h, w) = frame.shape
    i = 0
    j = 0
    while i < h:
        i_limit = min(i+size, h)
        while j < w:
            j_limit = min(j+size, w)
            region = frame[i:i_limit, j:j_limit]

            tvalue = np.min(region) + 105
            print(tvalue)
            frame[i:i_limit, j:j_limit] = np.array(threshold(Image.fromarray(region), tvalue))

            j = j_limit
        i = i_limit
        j = 0

    return Image.fromarray(frame)


def pipeline(image_path: Path, output_dir: Path, config: Iterable):
    output_path = output_dir / f"{image_path.stem}"

    frame = cv2.imread(str(image_path), 0)
    image = Image.fromarray(frame)

    if "global_hist" in config:
        image = histogram_equalization_global(image=image)
        output_path = Path(str(output_path) + "_ghe")
    if "local_hist" in config:
        image = histogram_equalization_local(image=image)
        output_path = Path(str(output_path) + "_lhe")
    if "gaussian" in config:
        radius = 1
        image = gaussian_filter(image=image, radius=radius)
        output_path = Path(str(output_path) + "_gn")
    if "canny" in config:
        image = canny_edge_detection(image=image)
        output_path = Path(str(output_path) + "_edges")
    if "ada_thresh" in config:
        image = adaptive_thresholding(image=image)
        output_path = Path(str(output_path) + "_adathresh")
    if "otsu" in config:
        image = otsu(image=image)
        output_path = Path(str(output_path) + "_otsu")
    if "median" in config:
        image = median_blur(image=image)
        output_path = Path(str(output_path) + "_md")

    output_path = str(output_path) + ".png"

    frame = np.array(image)
    cv2.imwrite(output_path, frame)

if __name__ == "__main__":
    image_path = results_path / "frames" / "FMR_015-TP1-1_converted" / "FMR_015-TP1-1_converted_frame0.png"

    output_dir = results_path / "pipeline_results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    configs = [
        ["global_hist", "local_hist", "ada_thresh", "median"]
    ]

    for config in configs:
        pipeline(image_path=image_path, output_dir=output_dir, config=config)