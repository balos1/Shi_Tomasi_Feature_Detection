Shi Tomasi Feature Detection
===

CUDA, OpenMP, and regular serial C implementations of Shi Tomasi feature detection. Done as a comparison of architecture project.

# How to Use

```
usage: ./stfd [-v,-vv] <full path to the image> [sigma] [windowsize] [num_features]
flags:
	-h: show this help menu
	-v: output basic execution information
	-vv: output all information... good for debugging
arguments:
	sigma: the sigma value for the Gaussian distribution used to form the convolution mask.
	windowsize: the size of a pixel 'neighborhood' in an image
	num_features: how many features to extract
```

# LICENSE

MIT