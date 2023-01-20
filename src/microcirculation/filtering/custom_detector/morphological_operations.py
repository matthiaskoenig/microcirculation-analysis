from PIL import Image
import numpy as np
import cv2


def dilate(image: Image, strel_size: tuple = (3, 3)) -> Image:
    image = np.array(image)
    strel = np.ones(strel_size, np.uint8)
    image = cv2.dilate(image, strel)
    return Image.fromarray(image)


def erode(image: Image, strel_size: tuple = (3, 3)) -> Image:
    image = np.array(image)
    strel = np.ones(strel_size, np.uint8)
    image = cv2.erode(image, strel)
    return Image.fromarray(image)


def opening(image: Image, strel_size: tuple = (3, 3)) -> Image:
    image = np.array(image)
    strel = np.ones(strel_size, np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, strel)
    return Image.fromarray(image)


def closing(image: Image, strel_size: tuple = (3, 3)) -> Image:
    image = np.array(image)
    strel = np.ones(strel_size, np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, strel)
    return Image.fromarray(image)

