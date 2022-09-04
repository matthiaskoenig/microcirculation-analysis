## change video framerate

https://trac.ffmpeg.org/wiki/How%20to%20speed%20up%20/%20slow%20down%20a%20video

## video stabilization
Simple video stabilization using OpenCV
http://nghiaho.com/?p=2093


1. Find the transformation from previous to current frame using optical flow for all frames. The transformation only consists of three parameters: dx, dy, da (angle). Basically, a rigid Euclidean transform, no scaling, no sharing.
2. Accumulate the transformations to get the “trajectory” for x, y, angle, at each frame.
3. Smooth out the trajectory using a sliding average window. The user defines the window radius, where the radius is the number of frames used for smoothing.
4. Create a new transformation such that new_transformation = transformation + (smoothed_trajectory – trajectory).
5. Apply the new transformation to the video.



https://adamspannbauer.github.io/python_video_stab
pip install vidstab[cv2]


# python video processing
https://abhitronix.github.io/vidgear/latest/gears/stabilizer/usage/
https://github.com/abhiTronix/vidgear

VidGear is a High-Performance Video Processing Python Library that provides an easy-to-use, highly extensible, thoroughly optimised Multi-Threaded + Asyncio API Framework on top of many state-of-the-art specialized libraries like OpenCV, FFmpeg, ZeroMQ, picamera, starlette, yt_dlp, pyscreenshot, aiortc and python-mss serving at its backend, and enable us to flexibly exploit their internal parameters and methods, while silently delivering robust error-handling and real-time performance

Hybrid Neural Fusion for Full-frame Video Stabilization
https://github.com/alex04072000/FuSta

Existing video stabilization methods often generate visible distortion or require aggressive cropping of frame boundaries, resulting in smaller field of views. In this work, we present a frame synthesis algorithm to achieve full-frame video stabilization. We first estimate dense warp fields from neighboring frames and then synthesize the stabilized frame by fusing the warped contents. Our core technical novelty lies in the learning-based hybrid-space fusion that alleviates artifacts caused by optical flow inaccuracy and fast-moving objects. We validate the effectiveness of our method on the NUS, selfie, and DeepStab video datasets. Extensive experiment results demonstrate the merits of our approach over prior video stabilization methods.
http://jiyang.fun/docs/jiyang_cvpr20.pdf