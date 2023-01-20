from typing import List
import cv2
from PIL import Image 
import numpy as np

from standard_transformations import *
from video_info import get_keypoints_for_frame
from morphological_operations import *
from utils import get_average_grayscale_value, stack_images

test_image_path = "/Users/maniklaldas/Desktop/sublingua.png"


def keypoint_detection(frame: Image, kp_method: str):
    frame = np.array(frame)
    return get_keypoints_for_frame(frame, kp_method)


def threshold_vessels_detection_avg_grayscale():
    """
    Detection of vessels by thresholding edge detected image using avg. grayscale value
    """
    image_original = Image.open(test_image_path).convert("L")
    image_edges = detect_edges(image_original)

    avg_grayscale_value = get_average_grayscale_value(image_edges)
    image_edges = threshold(image_edges, avg_grayscale_value)

    stack_images([image_original, image_edges])


def threshold_vessels_detection(threshold: int):
    """
    Detection of vessels by thresholding edge detected image using given threshold value
    """
    image_original = Image.open(test_image_path).convert("L")
    image = threshold(image_original, threshold)

    stack_images([image_original, image])


def morpho_closing_vessels_detection():
    """
    Isolation of vessels by applying closing operation on thresholded image
    """
    image_original = Image.open(test_image_path).convert("L")
    
    image0 = threshold(image_original, 120)
    image0 = np.array(closing(image0))

    image1 = threshold(image_original, 130)
    image1 = np.array(closing(image1))

    stack_images([image_original, Image.fromarray(image1)])


def blur_erosion_vessels_detection():
    """
    Isolation of vessels by applying erosion operation on gaussian blurred image
    """
    image_original = Image.open(test_image_path).convert("L")
    blur = gaussian_filter(image_original)
    
    image = Image.fromarray(np.array(image_original) - np.array(blur))
    
    image = erode(image)
    
    stack_images([image_original, image])


def morphological_vessels_detection():
    """
    Isolation of vessels by applying multiple morphological operations on the image
    """
    image_original = Image.open(test_image_path).convert("L")
    
    image = threshold(image_original, 130)
    image = closing(image)
    image = dilate(image)
    image = erode(image, (5, 5))

    stack_images([image_original, image])


if __name__ == "__main__":
    # input_image = ...
    morphological_vessels_detection()