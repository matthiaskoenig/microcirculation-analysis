"""
    The process calculates optical flow (``cv2.calcOpticalFlowPyrLK``) from frame to frame using
    keypoints generated by the keypoint method specified by the user.  The optical flow will
    be used to generate frame to frame transformations (``cv2.estimateRigidTransform``).
    Transformations will be applied (``cv2.warpAffine``) to stabilize video.


    This class is based on the `work presented by Nghia Ho <http://nghiaho.com/?p=2093>`_

    :param kp_method: String of the type of keypoint detector to use. Available options are:
                        ``["GFTT", "BRISK", "DENSE", "FAST", "HARRIS", "MSER", "ORB", "STAR"]``.
                        ``["SIFT", "SURF"]`` are additional non-free options available depending
                        on your build of OpenCV.  The non-free detectors are not tested with this package.

Dense Optical Flow

In this section, we will take a look at some Dense Optical Flow algorithms which can
calculate the motion vector for every pixel in the image.

Implementation
Since the OpenCV Dense Optical Flow algorithms have the same usage pattern, we’ve created the wrapper function for
convenience and code duplication avoiding.

At first, we need to read the first video frame and do image preprocessing if necessary:

"""
from pathlib import Path

import cv2
import numpy as np


# calc flow of movement
# optical_flow = cv2.calcOpticalFlowPyrLK(self.prev_gray,
#                                         current_frame_gray,
#                                         self.prev_kps, None)
from microcirculation import data_path


def dense_optical_flow(method, video_path, params=[], to_gray=False):
    """
    https://learnopencv.com/optical-flow-in-opencv/

    :param method:
    :param video_path:
    :param params:
    :param to_gray:
    :return:
    """
    # Read the video and first frame
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # The main part of the demo is a loop, where we calculate Optical Flow for each new pair of consecutive images.
    # After that, we encode the result into HSV format for visualization purposes:

    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break

        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

        # Update the previous frame
        old_frame = new_frame

"""
Therefore, this function reads two consecutive frames as method input. In some cases, the image grayscaling is needed, 
so the to_gray parameter should be set as True. After we got the algorithm output, we encode it for proper 
visualization using HSV color format.
"""

if __name__ == "__main__":
    video_stable: Path = data_path / "ops" / "output_stable.avi"

    # algorithm = 'lucaskanade_dense'
    algorithm = 'farneback'

    if algorithm == 'lucaskanade_dense':
        # performs interpolation
        method = cv2.optflow.calcOpticalFlowSparseToDense
        params = []
    elif algorithm == 'farneback':
        method = cv2.calcOpticalFlowFarneback
        params = [0.5, 3, 15, 3, 5, 1.2, 0]  # default Farneback's algorithm parameters

    frames = dense_optical_flow(
        method,
        video_path=str(video_stable),
        params=params,
        to_gray=True
    )
