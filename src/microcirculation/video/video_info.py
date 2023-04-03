"""Get video info."""
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd

# TODO: create metadata file and video overview
import cv2

def get_video_info(video_path: Path, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get video information from cv2."""

    if metadata is None:
        metadata: Dict[str, Any] = {}

    video_file = cv2.VideoCapture(str(video_path))
    frame_count: int = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height: int = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width: int = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_rate: int = int(video_file.get(cv2.CAP_PROP_FPS))
    duration: int = frame_count // frame_rate  # duration in seconds

    return {
        **metadata,
        "frame_height": frame_height,
        "frame_width": frame_width,
        "frame_count": frame_count,
        "frame_rate": frame_rate,
        "duration": duration,
    }


def get_video_infos(video_dir: List[Path], metadata: Optional[Dict[str, Any]] = None, xlsx_out: Optional[Path] = None) -> pd.DataFrame:
    """Overview of videos in directory."""

    data = []
    for p in video_dir.glob('**/*.avi'):
        if p.is_file():
            md = {
                "path": p.relative_to(video_dir),
                **metadata,
            }

            info = get_video_info(p, metadata=md)
            data.append(info)

    df = pd.DataFrame(data)

    # write to excel
    if xlsx_out:
        with pd.ExcelWriter(xlsx_out) as writer:
            df.to_excel(writer, sheet_name="videos", index=False)

    return df


if __name__ == "__main__":
    from microcirculation import data_dir
    from microcirculation.console import console

    video_example = data_dir / "test" / "FMR_010-TP1-1.avi"
    console.print(get_video_info(video_path=video_example))

    # The OPS has a resolution of approximately 1 µm/pixel (6)
    metadata = {
        "device": "ops",
        "pixel_width": 1,
        "pixel_height": 1,
        "pixel_unit": 'µm'
    }
    video_dir = data_dir / "rat_liver_ops"
    df = get_video_infos(
        video_dir=video_dir,
        metadata=metadata,
        xlsx_out=data_dir / "rat_liver_ops" / "video_info.xlsx",
    )
    console.print(df.to_string())
