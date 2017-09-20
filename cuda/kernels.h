/*
	File: kernels.h
    Author(s):
		Cody Balos - University of the Pacific, ECPE 293, Spring 2017
	Description:
    	GPU kernels and GPU kernel wrappers for CUDA Shi Tomasi Feature Detection program.
*/

#ifndef KERNELS_H
#define KERNELS_H

#include <stdio.h>
#include "data_wrapper_t.h"

// Wrap each cuda call in error checking macro when in debug mode.
// why: http://stackoverflow.com/tags/cuda/info
#ifdef DEBUG
#define gpu_err_chk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
#else
#define gpu_err_chk(ans) { ans; }
#endif

/// Check for error in cuda function call.
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
	  	exit(code);
   }
}

/// Does setup for cuda convolution, providing device pointers to the output horizontal and vertical gradients.
void cuda_convolve(float *image, int width, int height, float *gk, float *gkd, int k_width, int k_height, float **d_hgrad,
					         float **d_vgrad);

/// Does setup for cuda computation of eigenvalues of Z matrices, and launches the cuda kernel.
void cuda_compute_eigenvalues(float *d_hgrad, float *d_vgrad, int height, int width, int windowsize,
                              data_wrapper_t **d_eigenvalues);

/// Finds features based on the provided eigenvalues using the thrust CUDA library for some parts of the algorithm.
unsigned int cuda_find_features(data_wrapper_t *d_eigenvalues, int max_features, int image_width, int image_height,
                                data_wrapper_t **features);

#endif