from pathlib import Path

from PIL import Image

from microcirculation import resources_path, results_path
from microcirculation.filters.filter import *
from microcirculation.utils import stack_images


def apply_all_filters(image_path: Path, results_dir: Path) -> None:
    """Apply all filter pipelines to given image."""

    results_dir.mkdir(exist_ok=True, parents=True)

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
        # image_out = image_filtered
        # save image
        image_out_path = (
            results_dir
            / f"0{k+1}_{test_image_path.stem}_{f_filter_pipeline.__name__}.png"
        )

        image_out.save(str(image_out_path))


# FIXME: Also run all the pipelines:
# brightdess normalization
# preprocess_detect_vessel

if __name__ == "__main__":

    results_dir: Path = results_path / "filter_pipelines"
    test_image_path: Path = resources_path / "sublingua.png"
    apply_all_filters(image_path=test_image_path, results_dir=results_dir)

    # keypoint examples
    # TODO: Fix keypoints and plot keypoints on frames

    # keypoints_results_dir: Path = results_dir / "keypoints"
    # image_for_keypoints: Path = results_dir / "01_sublingua_threshold_vessels_detection.png"
    # superimpose_keypoints_on_image(image_path=image_for_keypoints, results_dir=keypoints_results_dir)
