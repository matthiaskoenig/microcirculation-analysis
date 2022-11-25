# Microcirculation analysis
Analysis of microcirculation videos from OPS and IDF devices.

## IO and conversion
### OPS 
OPS videos in the old format can be read via the CapiMetrics software and exported as AVI.


## Video processing with python

OpenCV python
https://github.com/opencv/opencv-python
pip install opencv-python


## change video framerate

https://trac.ffmpeg.org/wiki/How%20to%20speed%20up%20/%20slow%20down%20a%20video


# Stacking videos
Bonus: create a comparison video
Use the vstack or hstack filter, depending on if you want them stacked vertically or side-by-side:
ffmpeg -i clip.mkv -i clip-stabilized.mkv  -filter_complex vstack clips-stacked.mkv
ffmpeg -i FMR_015-TP3-2_converted.avi -i FMR_015-TP3-2_converted_stable.avi -filter_complex vstack -crf 20 FMR_015_TP3-2_stacked.avi
https://stackoverflow.com/questions/11552565/vertically-or-horizontally-stack-mosaic-several-videos-using-ffmpeg

# python video processing
https://abhitronix.github.io/vidgear/latest/gears/stabilizer/usage/
https://github.com/abhiTronix/vidgear

VidGear is a High-Performance Video Processing Python Library that provides an easy-to-use, highly extensible, thoroughly optimised Multi-Threaded + Asyncio API Framework on top of many state-of-the-art specialized libraries like OpenCV, FFmpeg, ZeroMQ, picamera, starlette, yt_dlp, pyscreenshot, aiortc and python-mss serving at its backend, and enable us to flexibly exploit their internal parameters and methods, while silently delivering robust error-handling and real-time performance

Hybrid Neural Fusion for Full-frame Video Stabilization
https://github.com/alex04072000/FuSta

Existing video stabilization methods often generate visible distortion or require aggressive cropping of frame boundaries, resulting in smaller field of views. In this work, we present a frame synthesis algorithm to achieve full-frame video stabilization. We first estimate dense warp fields from neighboring frames and then synthesize the stabilized frame by fusing the warped contents. Our core technical novelty lies in the learning-based hybrid-space fusion that alleviates artifacts caused by optical flow inaccuracy and fast-moving objects. We validate the effectiveness of our method on the NUS, selfie, and DeepStab video datasets. Extensive experiment results demonstrate the merits of our approach over prior video stabilization methods.
http://jiyang.fun/docs/jiyang_cvpr20.pdf