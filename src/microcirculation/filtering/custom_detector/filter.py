from typing import List
import cv2
from PIL import Image 
import numpy as np

from standard_filters import gaussian, gamma_transform
from video_info import get_keypoints_for_frame

test_image_path = "/Users/maniklaldas/Desktop/sublingua.png"

resize_dims = (300, 300)


def detect_edges(image: Image) -> Image:
    # RGB to gray
    image = image.convert("L")

    # Gaussian blur for denoising image
    image = gaussian(image)
    image = np.array(image)

    # histogram equalization
    image = cv2.equalizeHist(image)

    # image resizing
    # if image.shape[0] > resize_dims[0] or image.shape[1] > resize_dims[1]:
    #     image = cv2.resize(image, (min(image.shape[0], resize_dims[0]), min(image.shape[1], resize_dims[1])))

    # edge detection using Sobel filter
    edges = cv2.Sobel(src = image, ddepth = cv2.CV_64F, dx = 1, dy = 1, ksize = 3)

    return Image.fromarray(edges)


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


def get_average_grayscale_value(image: Image) -> int:
    image = np.array(image, dtype=np.uint8)
    return int(np.mean(image))


def stack_images(images: List[Image.Image]) -> None:
    widths, heights = zip(*(image.size for image in images))
    total_width = sum(widths)
    max_height = max(heights)

    stacked_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for image in images:
        stacked_image.paste(image, (x_offset,0))
        x_offset += image.size[0]

    stacked_image.save('final.png')
    stacked_image.show("final")


def keypoint_detection(frame: Image, kp_method: str):
    frame = np.array(frame)
    return get_keypoints_for_frame(frame, kp_method)


def custom1():
    image_original = Image.open(test_image_path).convert("L")
    image_edges = detect_edges(image_original)

    avg_grayscale_value = get_average_grayscale_value(image_edges)
    print(avg_grayscale_value)
    image_edges = threshold(image_edges, 20)

    stack_images([image_original, image_edges])


def custom2():
    image_original = Image.open(test_image_path).convert("L")
    image = threshold(image_original, 160)
    #image = erode(image)
    stack_images([image_original, image])


def custom3():
    image_original = Image.open(test_image_path).convert("L")
    # image = detect_edges(image_original)
    image0 = threshold(image_original, 120)
    image0 = np.array(closing(image0))

    image1 = threshold(image_original, 130)
    image1 = np.array(closing(image1))

    stack_images([image_original, Image.fromarray(image1)])


def custom4():
    image_original = Image.open(test_image_path).convert("L")
    blur = gaussian(image_original)
    image = Image.fromarray(np.array(image_original) - np.array(blur))
    image = erode(image)
    stack_images([image_original, image])


def custom5():
    image_original = Image.open(test_image_path).convert("L")
    
    image = threshold(image_original, 130)
    image = closing(image)
    image = dilate(image)
    image = erode(image, (5, 5))

    kp_frame = keypoint_detection(image, "SIFT") 

    stack_images([image_original, kp_frame])


if __name__ == "__main__":
    # input_image = ...
    custom5()