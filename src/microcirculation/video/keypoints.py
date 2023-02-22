import copy
from pathlib import Path
from enum import Enum
import cv2
import numpy as np
from PIL import Image

from typing import Iterable

from microcirculation.video.video_info import get_video_info


def keypoint_detection(image: Image.Image, kp_method: str) -> Image.Image:
    """Keypoint detection"""
    frame = np.array(image)
    return Image.fromarray(draw_keypoints_on_frame(frame, kp_method))



kp_methods = [
    "GFTT",
    "FAST",
    "ORB",
    "SURF",
    "SIFT",
]


# FIXME: create method to get the actual keypoints as array! (DONE)
# FIXME: use this information to plot it on frame & napari as layer (DONE)
def get_keypoints_for_frame(frame: np.ndarray, kp_method: str) -> np.ndarray:
    """
    Calculates keypoints for the image in the given frame and returns
    them as an array.

    @param: frame: the image in the form of a numpy array
    @param: kp_method: the keypoint detection method (these are standard detectors)
    """
    if kp_method not in kp_methods:
        raise ValueError(f"Unsupported Keypoint method: '{kp_method}'")

    if len(frame.shape) > 2:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    if kp_method == "GFTT":
        keypoints = np.int0(cv2.goodFeaturesToTrack(gray_frame, maxCorners=50, qualityLevel=0.05, minDistance=10))

    elif kp_method == "FAST":
        fast_detector = cv2.FastFeatureDetector_create()
        keypoints = fast_detector.detect(gray_frame, None)

    elif kp_method == "ORB":
        orb_detector = cv2.ORB_create(200, 2.0)
        keypoints, descriptor = orb_detector.detectAndCompute(gray_frame, None)

    elif kp_method == "SURF":
        surf_detector = cv2.xfeatures2d.SURF_create(50000)
        keypoints, descriptor = surf_detector.detectAndCompute(gray_frame, None)

    elif kp_method == "SIFT":
        sift_detector = cv2.SIFT_create()
        keypoints = sift_detector.detect(gray_frame, None)

    return keypoints


def draw_keypoints_on_frame(frame: np.ndarray, kp_method: str) -> np.ndarray:
    """
    Get Keypoints for given frame and superimposes the keypoints 
    on the frame in the form of circles.

    @param: frame: the image in the form of a numpy array
    @param: kp_method: the keypoint detection method (these are standard detectors)
    """

    keypoints = get_keypoints_for_frame(frame=frame, kp_method=kp_method)

    if len(frame.shape) > 2:
        keypoint_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        keypoint_frame = frame

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(keypoint_frame, (x, y), 3, 255, -1)

    return keypoint_frame


def superimpose_keypoints_on_image(image_path: Path, results_dir: Path) -> None:
    """
    Detect and superimpose keypoints on a single image (given by image path)
    
    @param: image_path: the path to the image for keypoint detection
    @param: results_dir: the directory in which the superimposed image is stored
    """
    results_dir.mkdir(exist_ok=True, parents=True)

    image: Image.Image = Image.open(image_path)
    keypoints_image: Image.Image = Image.fromarray(keypoint_detection(image, "SIFT"))

    keypoints_image_path = results_dir / f"{image_path.stem}_keypoints.png"
    keypoints_image.save(str(keypoints_image_path))


def get_keypoints_and_display(video_path: Path, kp_method: str):
    """
    Calculate keypoints and visualize on video/frames
    example: https://www.oreilly.com/library/view/computer-vision-with/9781788472395/1ff16b52-a319-4c94-b02d-574c56c84f75.xhtml
    """

    video_info = get_video_info(video_path)
    frame_rate = video_info["frame_rate"]

    video_in = cv2.VideoCapture(str(video_path))

    extension = str(video_path).split(".")[-1]
    outfile_path = "".join(str(video_path).split(".")[:-1]) + "_keypoints." + extension
    video_out = cv2.VideoWriter(
        outfile_path,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        frame_rate,
        (int(video_in.get(3)), int(video_in.get(4))),
    )

    while True:
        ret, frame = video_in.read()

        if ret:
            keypoint_frame = draw_keypoints_on_frame(frame, kp_method)
            video_out.write(keypoint_frame)
        else:
            break

    video_in.release()
    video_out.release()

    return outfile_path


def get_transparent_keypoint_frame(keypoints: Iterable, frame_size: Iterable):
    """
    (Should) return a frame with transparent background and keypoints on it

    Currently not working as expected (transparency doesn't seem working)
    """
    h, w = frame_size
    black_frame = np.zeros((h, w), dtype="uint8")
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(black_frame, (x, y), 3, (255, 255, 255))

    black_frame = Image.fromarray(black_frame)

    black_frame = black_frame.convert("RGBA")  # for alpha-conversion to make it transparent
    pixels = black_frame.getdata()
    new_pixels = []
    for item in pixels:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_pixels.append((0, 0, 0, 0))
        else:
            new_pixels.append(item)
 
    black_frame.putdata(new_pixels) # this frame is now transparent except for the keypoints
    return black_frame
    