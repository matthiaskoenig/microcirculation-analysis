import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from microcirculation import resources_dir, results_dir
from microcirculation.filters.standard_transformations import (
    normalize_frames_brightness,
)
from microcirculation.filters.vessel_detection import *
from microcirculation.utils import stack_images, write_frames_as_video


def apply_all_filters(image_path: Path, results_dir: Path) -> List[Image.Image]:
    """Apply all filter pipelines to given image."""

    results_dir.mkdir(exist_ok=True, parents=True)

    results_images = []
    for k, f_filter_pipeline in enumerate(
        [
            threshold_vessels_detection,
            threshold_vessels_detection_local,
            threshold_vessels_detection_avg_grayscale,
            morphological_vessels_detection,
            morpho_closing_vessels_detection,
            blur_erosion_vessels_detection,
        ]
    ):
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
        results_images.append(image_out)
        # image_out = image_filtered
        # save image
        image_out_path = (
            results_dir
            / f"0{k+1}_{test_image_path.stem}_{f_filter_pipeline.__name__}.png"
        )

        image_out.save(str(image_out_path))
    return results_images


if __name__ == "__main__":
    results_dir: Path = results_dir / "filter_pipelines"
    test_image_path: Path = resources_dir / "sublingua.png"
    apply_all_filters(image_path=test_image_path, results_dir=results_dir)
