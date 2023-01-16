from PIL import Image 
from PIL.ImageFilter import GaussianBlur
import cv2
import numpy as np

def gaussian(image: Image, radius: int = 1):
    """
    applies gaussian filter of kernel size (2r + 1, 2r + 1)
    """

    filtered_image = image.filter(GaussianBlur(radius=radius))

    return filtered_image

def gamma_transform(image: Image, gamma: float = 0.5):
    image = np.array(image)
    image = np.array(255*(image / 255) ** gamma, dtype = 'uint8')
    return Image.fromarray(image)