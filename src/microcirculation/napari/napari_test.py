import napari
import cv2

image_path = "/Users/maniklaldas/Desktop/sublingua.png"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

viewer = napari.view_image(image, rgb=False)
napari.run()