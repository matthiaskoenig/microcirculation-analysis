from pathlib import Path
import os
from typing import List

import numpy as np

from PIL import Image

from microcirculation import resources_path, results_path
from microcirculation.filters.vessel_detection import *
from microcirculation.utils import stack_images, write_frames_as_video
from microcirculation.filters.standard_transformations import normalize_frames_brightness


from microcirculation.video.keypoints import keypoint_detection


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


# FIXME: Also run all the pipelines:
# brightdess normalization
# preprocess_detect_vessel
# UPDATE: done
def run_frames_preprocessing(frames: np.array) -> np.array:
    """
    Run the video preprocessing pipeline on a set of frames

    @param: frames: array of 2D arrays representing the array of frames
    """

    for filter in [
        threshold_vessels_detection,
        threshold_vessels_detection_local,
        threshold_vessels_detection_avg_grayscale,
        morphological_vessels_detection,
        morpho_closing_vessels_detection,
        blur_erosion_vessels_detection,
    ]:
        # vessel detection using the filter
        for frame in frames:
            image = Image.fromarray(frame).convert("L")
            image_filtered = filter(image)
            frame = np.array(image_filtered)

        # inter-frame brightness normalization on filtered frames  
        normalized_frames = normalize_frames_brightness(frames = frames)

        write_frames_as_video(
            frames=normalized_frames,
            frame_size=normalized_frames[0].shape,
            frame_rate=20,
            video_out_path=Path(f"normalized_frames_{filter.__name__}.mp4")
        )


def apply_keypoint_detection_on_all_files_in_directory(dir_path: Path) -> None:
    for file in os.listdir(dir_path):
        file_path = dir_path / file
        results_dir = dir_path / "keypoints"
        keypoint_detection()


if __name__ == "__main__":

    results_dir: Path = results_path / "filter_pipelines"
    test_image_path: Path = resources_path / "sublingua.png"
    apply_all_filters(image_path=test_image_path, results_dir=results_dir)

    # keypoint examples


    # keypoints_results_dir: Path = results_dir / "keypoints"
    # image_for_keypoints: Path = results_dir / "01_sublingua_threshold_vessels_detection.png"
    # superimpose_keypoints_on_image(image_path=image_for_keypoints, results_dir=keypoints_results_dir)
