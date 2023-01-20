"""
https://www.paulirish.com/2021/video-stabilization-with-ffmpeg-and-vidstab/


**vidstab**
https://github.com/georgmartius/vid.stab#available-options-with-vidstab-filters


sudo apt-get install libvidstab-dev

# The first pass ('detect') generates stabilization data and saves to `transforms.trf`
# The `-f null -` tells ffmpeg there's no output video file

ffmpeg -i FMR_015-TP1-2_converted.avi -vf vidstabdetect -f null -



"""