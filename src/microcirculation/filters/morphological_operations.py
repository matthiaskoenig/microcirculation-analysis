"""Module defines the morphological operations."""
import cv2
import numpy as np
from PIL import Image

__all__ = [
    "dilate",
    "erode",
    "opening",
    "closing",
]


def dilate(image: Image.Image, strel_size: tuple = (3, 3)) -> Image.Image:
    """Dilate image."""
    image = np.array(image)
    strel = np.ones(strel_size, np.uint8)
    image = cv2.dilate(image, strel)
    return Image.fromarray(image)


def erode(image: Image.Image, strel_size: tuple = (3, 3)) -> Image.Image:
    """Erode image."""
    image = np.array(image)
    strel = np.ones(strel_size, np.uint8)
    image = cv2.erode(image, strel)
    return Image.fromarray(image)


def opening(image: Image.Image, strel_size: tuple = (3, 3)) -> Image.Image:
    """Opening image."""
    image = np.array(image)
    strel = np.ones(strel_size, np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, strel)
    return Image.fromarray(image)


def closing(image: Image.Image, strel_size: tuple = (3, 3)) -> Image.Image:
    """Close image."""
    image = np.array(image)
    strel = np.ones(strel_size, np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, strel)
    return Image.fromarray(image)
