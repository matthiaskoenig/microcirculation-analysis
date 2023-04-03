import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from microcirculation import results_dir
from microcirculation.utils import extract_video_frames, stringify_time

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
        keypoints = np.int0(
            cv2.goodFeaturesToTrack(
                gray_frame, maxCorners=50, qualityLevel=0.05, minDistance=10
            )
        )

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
