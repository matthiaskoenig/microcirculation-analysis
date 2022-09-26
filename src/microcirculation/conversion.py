from pathlib import Path
import numpy as np
import cv2 as cv


def ops_conversion(input_path: Path, output_path: Path):
    """Read OPS AVI and convert to correct avi."""

    cap = cv.VideoCapture(str(input_path))

    # Define the codec and create VideoWriter object
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # Most codecs are lossy. If you want lossless video file you need to use a
    # lossless codecs (eg. FFMPEG FFV1, Huffman HFYU, Lagarith LAGS, etc...)
    fourcc = cv.VideoWriter_fourcc(*'FFV1')  # lossless
    out = cv.VideoWriter(
        filename=str(output_path),
        fourcc=fourcc,
        fps=30.0,
        frameSize=(640, 480),
    )

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # TODO: process frames
        out.write(frame)

        # save results

        # Display results
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    from microcirculation import data_path
    input_path: Path = data_path / "FMR_015-TP1-1.avi"
    output_path: Path = data_path / "output.avi"
    print(input_path)
    ops_conversion(input_path=input_path, output_path=output_path)
