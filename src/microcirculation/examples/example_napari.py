"""Example running napari.

Read image and display.
"""

import napari
from pathlib import Path
from PIL import Image

from skimage import data
from skimage.io import imread
import cv2
from typing import Iterable

from microcirculation.video.keypoints import (
    get_keypoints_for_frame, 
    get_transparent_keypoint_frame,
    draw_keypoints_on_frame
)
from microcirculation import resources_path

def napari_example():
    """View image in napari.

    This is starting napari and blocking.
    """
    # viewer = napari.Viewer()
    cells = data.cells3d()[30, 1]  # grab some data
    viewer = napari.view_image(cells, colormap="magma")

    # image_path = data_path / "sublingua.png"
    # image_path = "/home/mkoenig/git/microcirculation-analysis/data/sublingua.png"
    # data = cv2.imread(str(image_path))
    # data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    # viewer.add_image(data)
    # viewer = napari.view_image(image, rgb=False)
    napari.run()


def superimpose_keypoints_on_frame(src_path: Path) -> None:
    frame = cv2.imread(str(src_path), 0) # 0 is for reading in grayscale format
    output_path = f"{src_path.parent}/{src_path.stem}_keypoints{src_path.suffix}"
    keypoint_frame = draw_keypoints_on_frame(frame=frame, kp_method="SIFT")
    cv2.imwrite(output_path, keypoint_frame)


def visualize_frames_in_napari(frames: Iterable):
    viewer = napari.Viewer()
    for frame in frames:
        viewer.add_image(frame)

    napari.run()

def visualize_frames_in_napari_from_path(frame_paths: Iterable):
    frames = []
    for path in frame_paths:
        frame = imread(str(path))
        frames.append(frame)
    
    visualize_frames_in_napari(frames=frames)

if __name__ == "__main__":
    # napari_example()
    # viewer = napari.Viewer()
    # napari.run()

    # image_path = resources_path / "sublingua.png"
    # get_keypoints_frame_and_superimpose(src_path=image_path)

    frame_paths = [
        resources_path / "sublingua.png",
        resources_path / "sublingua_keypoints.png"
    ]
    visualize_frames_in_napari_from_path(frame_paths=frame_paths)