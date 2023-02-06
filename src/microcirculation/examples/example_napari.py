"""Example running napari.

Read image and display.
"""

import napari
from skimage import data


def napari_example():
    """View image in napari.

    This is starting napari and blocking.
    """
    # viewer = napari.Viewer()
    cells = data.cells3d()[30, 1]  # grab some data
    viewer = napari.view_image(cells, colormap="magma")

    # image_path = data_path / "sublingua.png"
    # image_path = "/home/mkoenig/git/microcirculation-analysis/data/sublingua.png"
    # data = cv2.imread(str(image_path))
    # data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    # viewer.add_image(data)
    # viewer = napari.view_image(image, rgb=False)
    napari.run()


if __name__ == "__main__":
    napari_example()
    # viewer = napari.Viewer()
    # napari.run()
