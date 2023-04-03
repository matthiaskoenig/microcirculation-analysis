"""Code for video normalization.

Defines a list of preprocess functions which allows to simplify the normalization.
"""

from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image

from microcirculation import resources_dir, results_dir
from microcirculation.utils import extract_video_frames, write_frames_as_video
from microcirculation.video.keypoints import generate_keypoint_video
from microcirculation.video.stabilization import stabilize_video
from microcirculation.video.video_utils import generate_vessel_detected_video

# def apply_preprocess_filters(
#     video_in: Path, video_out_base: Path, filters=Iterable[Callable]
# ) -> Iterable[Path]:
#     """Apply preprocess filters for better video normalization."""
#     video_out_paths = []
#
#     # apply filters (for vessel detection)
#     for filter in filters:
#         video = cv2.VideoCapture(str(video_in))
#         video_out = f"{video_out_base.stem}_{filter.__name__}{video_out_base.suffix}"
#         print(video_out)
#         video_out_paths.append(Path(video_out))
#         video_out_buffer = cv2.VideoWriter(
#             video_out,
#             cv2.VideoWriter_fourcc(*"MJPG"),
#             int(video.get(cv2.CAP_PROP_FPS)),
#             (int(video.get(3)), int(video.get(4))),
#             False,
#         )
#
#         while True:
#             read_success, frame = video.read()
#
#             if read_success:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#                 image = Image.fromarray(frame)
#                 image = filter(image)
#                 # why conversion to uint8?
#                 frame = np.uint8(np.array(image))
#
#                 video_out_buffer.write(frame)
#             else:
#                 break
#
#         video_out_buffer.release()
#     return video_out_paths


if __name__ == "__main__":
    # --- Full pipeline on example video ---

    # convert

    # process video (vessel detection)

    # stabilization with keypoints from vessel structures

    # vessel processing

    # flow calculation

    video_path = resources_dir / "FMR_015-TP1-1_converted.avi"
    vessels_video_path = generate_vessel_detected_video(
        video_path=video_path, detection_config=["global_hist", "ada_thresh", "median"]
    )

    stabilize_video(
        original_video_path=video_path, vessels_video_path=vessels_video_path
    )
