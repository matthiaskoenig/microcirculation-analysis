from PIL import Image 
from PIL.ImageFilter import GaussianBlur
import cv2
import numpy as np


def gaussian_filter(image: Image, radius: int = 1):
    """
    applies gaussian filter of kernel size (2r + 1, 2r + 1)
    """
    filtered_image = image.filter(GaussianBlur(radius=radius))

    return filtered_image


def gamma_transform(image: Image, gamma: float = 0.5):
    image = np.array(image)
    image = np.array(255*(image / 255) ** gamma, dtype = 'uint8')
    return Image.fromarray(image)


def threshold(image: Image, tvalue: int, invert: bool = True) -> Image:
    """
    thresholds an image using given threshold value

                { 255, if I(x, y) >= tvalue
    I(x, y) =   {
                { 0, if I(x, y) < tvalue
    """

    image = np.array(image)
    _, thresholded_image = cv2.threshold(image, tvalue, 255, cv2.THRESH_BINARY)

    if invert:
        thresholded_image = np.invert(thresholded_image)

    return Image.fromarray(thresholded_image)


def detect_edges(image: Image) -> Image:
    """
    Edge detection in the image using Sobel filter
    """

    # RGB to gray
    image = image.convert("L")

    # Gaussian blur for denoising image
    image = gaussian_filter(image)
    image = np.array(image)

    # histogram equalization
    image = cv2.equalizeHist(image)

    # edge detection using Sobel filter
    edges = cv2.Sobel(src = image, ddepth = cv2.CV_64F, dx = 1, dy = 1, ksize = 3)

    return Image.fromarray(edges)