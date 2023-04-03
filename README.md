# Microcirculation analysis
Analysis of microcirculation videos from OPS and IDF devices.

## Installation
Make virtualenv with python 3.10
```
mkvirtualenv microcirculation --python=python3.10
(microcirculation) pip install -r requirements.txt
```

## OPS Analysis
1. Create overview of videos with [video/video_info.py](./src/microcirculation/video/video_info.py)
2. Convert OPS videos to new AVI format with [ops/video_conversion.py](./src/microcirculation/ops/video_conversion.py)



© 2022-2023 Sankha Das & Matthias König