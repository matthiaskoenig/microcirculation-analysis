"""
1. Get information for given video (use cv library)
=>
"""
from pathlib import Path
import cv2
import os
import subprocess
import numpy as np
import copy

composite_videos_path = Path("./composite_videos")


def get_video_info(video_path: Path):
    """
    1. Get video meta data
        frame_height: int
        frame_width: int
        frame_rate: int  # frames per second
        pixel_width: float  # [µm] (magnification, ... spacial resolution; -> dictionary with defintions; OPS_10x =; OPS_5x ... IDF = ...)
        pixel_height: float  # [µm] not always square
    """

    video_file = cv2.VideoCapture(str(video_path))
    frame_count: int = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height: int = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width: int = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_rate: int = int(video_file.get(cv2.CAP_PROP_FPS))
    duration: int = frame_count // frame_rate  # duration in seconds

    pixel_width = 0  # TODO: need to figure out
    pixel_height = 0  # TODO: need to figure out

    return {
        "frame_count": frame_count,
        "frame_rate": frame_rate,
        "duration": duration,
        "frame_height": frame_height,
        "frame_width": frame_width,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
    }


def get_composite_video(
    video_1_path: Path, video_2_path: Path, alignment: str = "horizontal"
):
    """
    2.
    Create composite video: unstabilized/stabilized;
    => make video double the width: with left
    => clue 2 videos:
    call ffmpeg from python:
    => create example for: IDF sublingual, IDF rat, OPS rat (status quo)
    """

    assert alignment == "vertical" or alignment == "horizontal"

    video_1_info = get_video_info(video_1_path)
    print(video_1_info)
    video_2_info = get_video_info(video_2_path)
    print(video_2_info)

    if "composite_videos" not in os.listdir("./"):
        os.mkdir("./composite_videos")

    composite_file_name = (
        alignment
        + "_"
        + str(video_1_path).replace("/", "_")
        + "_"
        + str(video_2_path).replace("/", "_")
    )
    composite_file_path = composite_videos_path / composite_file_name

    if alignment == "horizontal":
        assert video_1_info["frame_height"] == video_2_info["frame_height"]
        subprocess.call(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_1_path),
                "-i",
                str(video_2_path),
                "-filter_complex",
                "hstack=inputs=2",
                str(composite_file_path),
            ]
        )
    else:
        assert video_1_info["frame_width"] == video_2_info["frame_width"]
        subprocess.call(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_1_path),
                "-i",
                str(video_2_path),
                "-filter_complex",
                "vstack=inputs=2",
                str(composite_file_path),
            ]
        )

    video_file = cv2.VideoCapture(str(composite_file_path))
    while video_file.isOpened():
        ret, frame = video_file.read()
        if ret:
            cv2.imshow(composite_file_name, frame)
            cv2.setWindowProperty(composite_file_name, cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            break

    video_file.release()
    cv2.destroyAllWindows()


def get_keypoints_for_frame(frame, kp_method: str):
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


def get_keypoints_and_dislay(video_path: Path, kp_method: str):
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
            keypoint_frame = get_keypoints_for_frame(frame, kp_method)
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


# video_path = Path("/Users/maniklaldas/Desktop/BRM-TC-Jena-P0-AdHoc-1-20220901-092449047---V0.avi")
video_path = Path("/Users/maniklaldas/Desktop/FMR_015-TP1-1_converted.avi")

keypoint_video_path = Path(get_keypoints_and_dislay(video_path, "SIFT"))

# get_composite_video(video_path, keypoint_video_path, "horizontal")


"""
3. Overview table of the videos
=> excel spreadsheet: especially important for the OPS videos; what degree of hepatectomy
=> Matthias: mapping of video ids to biological information; 
species: human/rat
tissue: liver/sublingual
"""
