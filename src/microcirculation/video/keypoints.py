import copy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from microcirculation.video.video_info import get_video_info


def keypoint_detection(frame: Image, kp_method: str):
    """Keypoint detection"""
    frame = np.array(frame)
    return draw_keypoints_on_frame(frame, kp_method)


def superimpose_keypoints_on_image(image_path: Path, results_dir: Path) -> None:

    results_dir.mkdir(exist_ok=True, parents=True)

    image: Image.Image = Image.open(image_path)
    keypoints_image: Image.Image = Image.fromarray(keypoint_detection(image, "SIFT"))

    keypoints_image_path = results_dir / f"{image_path.stem}_keypoints.png"
    keypoints_image.save(str(keypoints_image_path))


# FIXME: create method to get the actual keypoints as array!
# FIXME: use this information to plot it on frame & napari as layer


def draw_keypoints_on_frame(frame: np.ndarray, kp_method: str):
    """Get Keypoints for given frame."""

    # convert to grey scale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if kp_method == "HARRIS":
        harris_kp = cv2.cornerHarris(gray_frame, 2, 3, 0.04)

        threshold_fraction: float = 0.01

        keypoints_frame = np.array(copy.copy(gray_frame))
        keypoints_frame[harris_kp > threshold_fraction * harris_kp.max()] = [255, 0, 0]

        return keypoints_frame

    elif kp_method == "GFTT":
        gftt_kp = np.int0(cv2.goodFeaturesToTrack(gray_frame, 50, 0.05, 10))

        for i in gftt_kp:
            x, y = i.ravel()
            cv2.circle(gray_frame, (x, y), 3, 255, -1)

        return gray_frame

    elif kp_method == "FAST":
        num_keypoints: int = 100
        fast_detector = cv2.FastFeatureDetector_create(num_keypoints)
        fast_kp = fast_detector.detect(gray_frame, None)
        frame_kp = cv2.drawKeypoints(gray_frame, fast_kp, None, flags=0)

        return frame_kp

    elif kp_method == "ORB":
        orb = cv2.ORB_create(200, 2.0)
        keypoints, descriptor = orb.detectAndCompute(gray_frame, None)

        keypoints_frame = copy.copy(gray_frame)
        cv2.drawKeypoints(
            gray_frame,
            keypoints,
            keypoints_frame,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        print("Number of keypoints = " + str(len(keypoints)))
        return keypoints_frame

    elif kp_method == "SURF":
        surf = cv2.xfeatures2d.SURF_create(50000)
        keypoints, descriptor = surf.detectAndCompute(gray_frame, None)

        keypoints_frame = cv2.drawKeypoints(gray_frame, keypoints, None, (255, 0, 0), 4)
        print("Number of keypoints = " + str(len(keypoints)))

        return keypoints_frame

    elif kp_method == "SIFT":
        sift = cv2.SIFT_create()
        keypoints = sift.detect(gray_frame, None)
        keypoints_frame = cv2.drawKeypoints(gray_frame, keypoints, gray_frame)
        # keypoints_frame = cv2.drawKeypoints(gray_frame, keypoints, gray_frame, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print("Number of keypoints = " + str(len(keypoints)))
    return keypoints_frame


def get_keypoints_and_display(video_path: Path, kp_method: str):
    """
    3. Calculate keypoints and visualize on video/frames
    example: https://www.oreilly.com/library/view/computer-vision-with/9781788472395/1ff16b52-a319-4c94-b02d-574c56c84f75.xhtml
    """

    video_info = get_video_info(video_path)
    frame_rate = video_info["frame_rate"]
    frame_size = (video_info["frame_height"], video_info["frame_width"])

    video_in = cv2.VideoCapture(str(video_path))

    extension = str(video_path).split(".")[-1]
    outfile_path = "".join(str(video_path).split(".")[:-1]) + "_keypoints." + extension
    video_out = cv2.VideoWriter(
        outfile_path,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        frame_rate,
        (int(video_in.get(3)), int(video_in.get(4))),
    )

    kp_frames = []
    while True:
        ret, frame = video_in.read()

        if ret:
            keypoint_frame = draw_keypoints_on_frame(frame, kp_method)
            kp_frames.append(keypoint_frame)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            break

    for frame in kp_frames:
        video_out.write(frame)

    video_in.release()
    video_out.release()

    return outfile_path
