# TODO
## example frames
- [x] write code to extract frames from videos
- [x] create a set of test images from videos covering the different tests (IPS_sublingual, IPS_liver, OPS_liver, IDF_liver)

## add smaller steps in processing (with example output)
- [x] global equalization (contrast enhancement)
- [x] local histogram equalization adjustment (already implemented)

- [~] edge detection
- [~] otsu thresholding;


## write custom keypoint method
- [ ] process single frames and return dummy keypoint data structure which can be used in stabilization
  - Combination of filters, be careful with radius of filters which has to match to filter
    - [ ] apply a Gaussian filter (denoising the image); the kernel size of the filter has to fit to the vessel geometrie (small enough to not remove structure and big enough to remove noise; ~1µm diameter). => we are interested in structures > 3µm;
    - [ ] histogram normalizion [0, 255]
    - [ ] reduce frame size (make smaller, less pixels)
    - [ ] apply an edge detection filter (second order filter, Soebel, should be linear)
    - [~] we want nice edges, so perhaps apply a morphological filter to grow areas (dilatation filter)
    - [ ] find keypoints in the edges (sweet spot: enough keypoints, but not too many; perhaps stars of connections)
    - [ ] tresholding
  
    => alternatively use a custom filter for the vessels/tubes (Frangi filters)
  - Calculate keypoints on processed frame consisting of core vessel structure
  