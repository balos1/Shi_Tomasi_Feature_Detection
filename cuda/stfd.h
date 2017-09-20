/*
	File: cudaprogram.h
    Author(s): 
		Yang Liu - University of the Pacific, ECPE 293, Spring 2017
		Cody Balos - University of the Pacific, ECPE 293, Spring 2017
	Description:
    	Declares the functions used for the serial Shi Tomasi feature detection program.
*/

#ifndef SHI_TOMASI_H
#define SHI_TOMASI_H

#include "data_wrapper_t.h"

#define BUFFER 512

/// Draws a box at specfied location in the image. Used for markgin features.
void draw_features(data_wrapper_t *features, unsigned int count, float *image, int image_width, int image_height);

// Sort data_wrapper_t types by their index (x first then y) in ascending order.
int sort_data_wrapper_index_asc(const void *a, const void *b);

/// Creates Gaussian kernel and Gaussian derivative kernel for image gradient/convolution procedure.
void gen_kernel(float *gkernel, float *dkernel, float sigma, int a, int w);

/// Prints out the program help menu.
void help(const char *err);

/// Print out all of the detected features.
void print_features(data_wrapper_t *features, unsigned int count);

#endif