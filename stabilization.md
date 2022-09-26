# Video stabilization
An important step in the analysis is the stabilization of the video.
This will likely work more robust on a pre-processed video (smoothing & thresholding)

A curated list of stabilization methods is available:
https://github.com/yaochih/awesome-video-stabilization





## Optical flow
https://www.youtube.com/watch?v=5AUypv5BNbI
https://www.youtube.com/watch?v=4v_keMNROv4
https://www.codespeedy.com/optical-flow-in-opencv-python/
https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
https://github.com/pathak22/pyflow

https://learnopencv.com/optical-flow-in-opencv/
Optical flow is a task of per-pixel motion estimation between two consecutive frames in one video. Basically, the Optical Flow task implies the calculation of the shift vector for pixel as an object displacement difference between two neighboring images. The main idea of Optical Flow is to estimate the object’s displacement vector caused by it’s motion or camera movements.

There are two types of Optical Flow, and the first one is called **Sparse Optical Flow**. It computes the motion vector for the specific set of objects (for example – detected corners on image).
Hence, it requires some preprocessing to extract features from the image, which will be the basement for the Optical Flow calculation. OpenCV provides some algorithm implementations to solve the Sparse Optical Flow task:
- only for objects


Using only a sparse feature set means that we will not have the motion information about pixels that are not contained in it. This restriction can be lifted using Dense Optical Flow algorithms which are supposed to calculate a motion vector for every pixel in the image. Some Dense Optical Flow algorithms are already implemented in OpenCV: