from typing import Iterable

import napari

viewer = None


def get_napari_viewer():
    """Returns napari viewer instance or creates new one if not existing."""
    if viewer:
        return viewer
    else:
        return napari.Viewer()


def visualize_frame_and_keypoints(frame, keypoints):
    """Visualize frame and keypoints in napari."""
    viewer = napari.Viewer()
    viewer.add_image(frame)


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
