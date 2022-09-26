"""
https://www.paulirish.com/2021/video-stabilization-with-ffmpeg-and-vidstab/


**vidstab**
https://github.com/georgmartius/vid.stab#available-options-with-vidstab-filters


sudo apt-get install libvidstab-dev

# The first pass ('detect') generates stabilization data and saves to `transforms.trf`
# The `-f null -` tells ffmpeg there's no output video file

ffmpeg -i FMR_015-TP1-2_converted.avi -vf vidstabdetect -f null -


# Stacking videos
Bonus: create a comparison video
Use the vstack or hstack filter, depending on if you want them stacked vertically or side-by-side:
ffmpeg -i clip.mkv -i clip-stabilized.mkv  -filter_complex vstack clips-stacked.mkv
ffmpeg -i FMR_015-TP3-2_converted.avi -i FMR_015-TP3-2_converted_stable.avi -filter_complex vstack -crf 20 FMR_015_TP3-2_stacked.avi
https://stackoverflow.com/questions/11552565/vertically-or-horizontally-stack-mosaic-several-videos-using-ffmpeg
"""