"""
Very similar opencv approach.

https://github.com/sartorius-research/video_stabilizer/blob/master/video_stabilizer/stab.py

"""

frame_generator = video_stabilizer.generate_frames_from_video(cap)

for stabilized_frame, transform in video_stabilizer.stabilize_video(frame_generator):
    do_something(stabilized_frame, transform)
