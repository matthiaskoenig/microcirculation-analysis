import copy
from pathlib import Path
import os
from typing import Iterable
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from microcirculation.utils import extract_video_frames
from microcirculation import results_dir
from microcirculation.utils import stringify_time

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


def generate_keypoint_video(video_path: Path, kp_method: str = "SIFT") -> Path:
    """
    Calculate keypoints and visualize on video/frames
    example: https://www.oreilly.com/library/view/computer-vision-with/9781788472395/1ff16b52-a319-4c94-b02d-574c56c84f75.xhtml
    """

    start_time = datetime.now()

    if "keypoint_videos" not in os.listdir(results_dir):
        os.mkdir(results_dir / "keypoint_videos")
    keypoint_video_path = results_dir / "keypoint_videos" / f"{video_path.stem}_keypoints{video_path.suffix}"

    video_frames, frame_size, frame_rate = extract_video_frames(video_path)

    video_out = cv2.VideoWriter(
        str(keypoint_video_path),
        cv2.VideoWriter_fourcc(*'MJPG'),
        frame_rate,
        frame_size,
        False
    )

    for frame in video_frames:
        keypoint_frame = draw_keypoints_on_frame(frame, kp_method)
        video_out.write(keypoint_frame)

    video_out.release()

    end_time = datetime.now()

    keypoint_detection_time = int((end_time - start_time).total_seconds())
    print(f"*** Keypoints detected in {stringify_time(keypoint_detection_time)} ***")    

    return keypoint_video_path


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
