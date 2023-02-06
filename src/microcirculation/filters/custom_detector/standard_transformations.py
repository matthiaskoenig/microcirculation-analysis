import numpy as np
import cv2
from PIL import Image
from PIL.ImageFilter import GaussianBlur
from typing import Iterable

__all__ = [
    "gaussian_filter",
    "gamma_transform",
    "threshold",
    "global_histogram_equalization",
    "local_histogram_equalization",
    "median_blur",
    "detect_edges_sobel",
    "normalize_frames_brightness"
]


def gaussian_filter(image: Image, radius: int = 1) -> Image:
    """Apply gaussian filter of kernel size (2r + 1, 2r + 1)."""
    return image.filter(GaussianBlur(radius=radius))


def gamma_transform(image: Image, gamma: float = 0.5) -> Image:
    """Apply gamma transformation."""
    image = np.array(image)
    image = np.array(255 * (image / 255) ** gamma, dtype="uint8")
    return Image.fromarray(image)


def global_histogram_equalization(image: Image) -> Image:
    image_matrix = np.array(image)
    return Image.fromarray(cv2.equalizeHist(image_matrix))


def local_histogram_equalization(image: Image) -> Image:
    """Apply local histogram equalization using CLAHE"""
    image_matrix = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return Image.fromarray(clahe.apply(image_matrix))
    

# def box_filter(image: Image) -> Image:
#     image_matrix = np.array(image)


def median_blur(image: Image) -> Image:
    image_matrix = np.array(image)
    return Image.fromarray(cv2.medianBlur(image_matrix, 3))


def threshold(image: Image, value: int, invert: bool = False) -> Image:
    """Threshold an image using given threshold value.

                { 255, if I(x, y) >= tvalue
    I(x, y) =   {
                { 0, if I(x, y) < tvalue

    :param value: treshold value
    :param invert: invert the image
    """

    image = np.array(image)
    _, thresholded_image = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)

    if invert:
        thresholded_image = np.invert(thresholded_image)

    return Image.fromarray(thresholded_image)


def detect_edges_sobel(image: Image) -> Image:
    """Edge detection in the image using Sobel filter."""

    # RGB to gray
    image = image.convert("L")

    # Gaussian blur for denoising image
    image = gaussian_filter(image)
    image = np.array(image)

    # histogram equalization
    image = cv2.equalizeHist(image)

    # edge detection using Sobel filter
    edges = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)

    return Image.fromarray(edges)

def normalize_frames_brightness(frames: Iterable[np.array]):
    max_intensity = np.max(frames)
    avg_intensity = np.mean(frames)
    frames = frames/max_intensity
    frames = (frames*avg_intensity)

    # print(frames)
    return frames
