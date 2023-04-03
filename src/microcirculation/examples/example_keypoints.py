"""Example running napari.

Read image and display.
"""
from pathlib import Path
from typing import Iterable

import cv2
import napari
import numpy as np
from PIL import Image
from skimage import data
from skimage.io import imread

from microcirculation import resources_dir, results_dir
from microcirculation.filters.vessel_detection import threshold_vessels_detection_local
from microcirculation.video.keypoints import (
    draw_keypoints_on_frame,
    generate_keypoint_video,
    get_keypoints_for_frame,
)


def superimpose_keypoints_on_frame(src_path: Path) -> None:
    """Superimpose keypoints on frame and save image."""

    frame = cv2.imread(str(src_path), 0)  # 0 is for reading in grayscale format
    output_path = f"{src_path.parent}/{src_path.stem}_keypoints{src_path.suffix}"
    keypoint_frame = draw_keypoints_on_frame(frame=frame, kp_method="SIFT")
    cv2.imwrite(output_path, keypoint_frame)


if __name__ == "__main__":
    # napari_example()
    # viewer = napari.Viewer()
    # napari.run()
    # # superimpose_keypoints_on_frame(src_path=image_path)
    #
    # frame_paths = [
    #     resources_path / "sublingua.png",
    #     resources_path / "sublingua_keypoints.png"
    # ]
    # # visualize_frames_in_napari_from_path(frame_paths=frame_paths)

    # from microcirculation.napari_visualization import get_napari_viewer

    # viewer = get_napari_viewer()
    # image_path = resources_path / "sublingua.png"
    # frame = cv2.imread(str(image_path))
    # viewer.add_image(frame, name="rgb image", visible=False)
    # frame_gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # viewer.add_image(frame_gray, name="gray image")

    # # frame preprocessing

    # image_gray = Image.fromarray(frame_gray)
    # image_processed = threshold_vessels_detection_local(image_gray)
    # frame_processed = np.array(image_processed)
    # viewer.add_image(frame_processed, name="processed image", opacity=0.5)

    # # keypoint detection and visualization
    # keypoints: Iterable[cv2.KeyPoint] = get_keypoints_for_frame(frame_processed, kp_method="SIFT")
    # # print(keypoints)
    # kp: cv2.KeyPoint
    # points = [(kp.pt[1], kp.pt[0]) for kp in keypoints]

    # # points = np.array([[100, 100], [200, 200], [300, 100]])
    # points_layer = viewer.add_points(points, opacity=0.5, edge_width=0.2, face_color="white", edge_color="black", size=10)

    # napari.run()

    video_path = (
        results_dir
        / "vessel_videos"
        / "BRM-TC-Jena-P0-AdHoc-1-20220901-092449047---V0_vessels..avi"
    )

    keypoint_video_path = generate_keypoint_video(video_path=video_path)
