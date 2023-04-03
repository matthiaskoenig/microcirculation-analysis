from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from PIL.ImageFilter import GaussianBlur

__all__ = [
    "gaussian_filter",
    "gamma_transform",
    "threshold",
    "histogram_equalization_global",
    "histogram_equalization_local",
    "median_blur",
    "detect_edges_sobel",
    "normalize_frames_brightness",
    "canny_edge_detection",
    "adaptive_thresholding",
    "otsu",
]


def gaussian_filter(image: Image, radius: int = 1) -> Image:
    """Apply gaussian filter of kernel size (2r + 1, 2r + 1)."""
    return image.filter(GaussianBlur(radius=radius))


def gamma_transform(image: Image, gamma: float = 0.5) -> Image:
    """Apply gamma transformation."""
    image = np.array(image)
    image = np.array(255 * (image / 255) ** gamma, dtype="uint8")
    return Image.fromarray(image)


def median_blur(image: Image) -> Image:
    """Apply median blur.

    The function smoothes an image using the median filter with the ksize×ksize aperture.
    Each channel of a multi-channel image is processed independently. In-place operation is supported.
    """
    frame = np.array(image)
    frame_blur = cv2.medianBlur(frame, ksize=5)
    return Image.fromarray(frame_blur)


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


def histogram_equalization_global(image: Image) -> Image:
    """Global histogram equalization of single frame.

    Requires greyscale imge.
    It is a method that improves the contrast in an image, in order to stretch out the intensity range
    """
    frame = np.array(image)
    frame_equalized = cv2.equalizeHist(frame)
    return Image.fromarray(frame_equalized)


def histogram_equalization_local(image: Image) -> Image:
    """Apply local histogram equalization using CLAHE.

    https://docs.opencv.org/3.4/d6/db6/classcv_1_1CLAHE.html

        clipLimit: This is the threshold for contrast limiting
        tileGridSize: Divides the input image into M x N tiles and then applies histogram equalization
        to each local tile
    """
    frame = np.array(image)
    # print(type(frame))
    # print(frame.dtype)  # uint8: 0 to 255
    # print("Dimensions: ", frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    # FIXME: this is not working;

    return Image.fromarray(clahe.apply(frame))


def normalize_frames_brightness(frames: Iterable[np.array]) -> Iterable[np.array]:
    """Normalizes the brightness of multiple frames.

    FIXME: check this
    """
    assert frames != None
    frames = np.array(frames)
    max_intensity = np.max(frames)
    avg_intensity = np.mean(frames)
    frames = frames / max_intensity
    frames = frames * avg_intensity

    return frames


def canny_edge_detection(image: Image.Image):
    frame = np.array(image)
    edges = cv2.Canny(frame, 100, 200)
    return Image.fromarray(edges)


def adaptive_thresholding(image: Image.Image):
    frame = np.array(image)
    frame = cv2.adaptiveThreshold(
        frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 2
    )
    return Image.fromarray(frame)


def otsu(image: Image.Image):
    frame = np.array(image)
    blur = cv2.GaussianBlur(frame, (25, 25), 0)
    succ_value, frame = cv2.threshold(
        frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return frame
