from PIL import Image
import numpy as np
from pathlib import Path
import cv2
from typing import Iterable

from filter import *
from standard_transformations import normalize_frames_brightness

def preprocess_detect_vessel(video_path: Path):
    video = cv2.VideoCapture(str(video_path))

    extension = str(video_path).split(".")[-1]
    outfile_path = "".join(str(video_path).split(".")[:-1]) + "_vessels." + extension
    video_out = cv2.VideoWriter(
        outfile_path,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        int(video.get(cv2.CAP_PROP_FPS)),
        (int(video.get(3)), int(video.get(4))),
    )

    vessel_frames = []
    while True:
        ret, frame = video.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = Image.fromarray(frame)
            detected_vessels = threshold_vessels_detection_local(frame)
            vessel_frames.append(np.array(detected_vessels))

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            break

    for frame in vessel_frames:
        video_out.write(frame)

    return outfile_path


def normalize_video_brightness(video_path: Path):
    video = cv2.VideoCapture(str(video_path))
    
    extension = str(video_path).split(".")[-1]
    outfile_path = "".join(str(video_path).split(".")[:-1]) + "_normalized." + extension
    video_out = cv2.VideoWriter(
        outfile_path,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        int(video.get(cv2.CAP_PROP_FPS)),
        (int(video.get(3)), int(video.get(4))),
    )

    frames = []
    while True:
        ret, frame = video.read()

        if ret:
            frames.append(frame)
        else:
            break

    normalized_frames = normalize_frames_brightness(frames=frames)

    for frame in normalized_frames:
        frame = np.uint8(frame)
        video_out.write(frame)

    return outfile_path


def detect_vessels_and_display(video_path: Path):
    vessel_detected_video_path = preprocess_detect_vessel(video_path=video_path)
    print("Vessels detected")
    normalized_brightness_video_path = normalize_video_brightness(video_path=vessel_detected_video_path)
    print(normalized_brightness_video_path)


video_path = "/Users/maniklaldas/Desktop/BRM-TC-Jena-P0-AdHoc-1-20220901-092449047---V0.avi"
detect_vessels_and_display(video_path=video_path)
    
