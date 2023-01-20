from pathlib import Path
from typing import Tuple

import numpy as np
import cv2 as cv


def ops_conversion(
    input_path: Path,
    output_path: Path,
    fps_out: float,
    frame_size: Tuple[int, int],
    show: bool = True,
):
    """Read OPS AVI and convert to correct avi."""

    cap = cv.VideoCapture(str(input_path))

    # Define the codec and create VideoWriter object
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # Most codecs are lossy. If you want lossless video file you need to use a
    # lossless codecs (eg. FFMPEG FFV1, Huffman HFYU, Lagarith LAGS, etc...)
    fourcc = cv.VideoWriter_fourcc(*"FFV1")  # lossless
    out = cv.VideoWriter(
        filename=str(output_path),
        fourcc=fourcc,
        fps=fps_out,
        frameSize=frame_size,
    )
    from matplotlib import pyplot as plt

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # print(type(frame))
        # print(frame.shape)
        #
        # # convert to gray scale
        # image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # plt.subplot(121)
        # plt.imshow(image[:, :], cmap="gray", aspect="equal")
        # plt.axis('off')
        #
        # plt.subplot(122)
        # hist, bins = np.histogram(image[:, :], bins=256)  # , 256, [0,256])
        # bin_width = bins[1] - bins[0]
        # plt.bar(bins[:-1], hist * bin_width, bin_width, color="black")
        # plt.show()

        # TODO: process frames
        out.write(frame)

        # Display results
        if show:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imshow("frame", gray)
            if cv.waitKey(1) == ord("q"):
                break
    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    from microcirculation import data_path

    input_path: Path = data_path / "ops" / "FMR_015-TP1-2.avi"
    output_path: Path = data_path / "ops" / "FMR_015-TP1-2_converted.avi"
    ops_conversion(
        input_path=input_path,
        output_path=output_path,
        fps_out=30.0,
        frame_size=(640, 480),
        show=True,
    )

    # convert ops videos
    # input_dir: Path = data_path / "ops"
    # for video_in in sorted(input_dir.glob('*.avi')):
    #     if not "converted" in str(video_in):
    #         print(video_in)
    #         video_out = video_in.parent / f"{video_in.stem}_converted.avi"
    #         ops_conversion(
    #             input_path=video_in,
    #             output_path=video_out,
    #             fps_out=30.0,
    #             frame_size=(640, 480),
    #             show=False,
    #         )

    # input_path: Path = data_path / "braedius" / "BRM-TC-Jena-P3-AdHoc-3-20220901-113654379---V0.avi"
    # output_path: Path = data_path / "braedius" / "output.avi"

    # print(input_path)
    # ops_conversion(
    #     input_path=input_path,
    #     output_path=output_path,
    #     fps_out=30.0,
    #     frame_size=(1772, 1328),
    # )
