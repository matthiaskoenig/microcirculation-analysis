from PIL import Image 
import numpy as np
from typing import List

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