/*
	File: serialprogram.h
    Author(s):
		Yang Liu - University of the Pacific, ECPE 293, Spring 2017
		Cody Balos - University of the Pacific, ECPE 293, Spring 2017
	Description:
    	Declares the functions used for the serial Shi Tomasi feature detection program.
*/

#ifndef SHI_TOMASI_H
#define SHI_TOMASI_H

typedef struct data_wrapper_t {
	float data;
	int x;
	int y;
} data_wrapper_t;

/// Draws a box at specfied location in the image. Used for markgin features.
void draw_features(data_wrapper_t *features, unsigned int count, float *image, int image_width, int image_height);

/// Find features in an image.
unsigned int find_features(data_wrapper_t *eigenvalues, int max_features, int image_width, int image_height, data_wrapper_t **features);

/// Defines comparison for data_wrapper_t. When used with qsort, it will result in a descending order array.
int sort_data_wrapper_value_desc(const void *a, const void *b);

// Sort data_wrapper_t types by their index (x first then y) in ascending order.
int sort_data_wrapper_index_asc(const void *a, const void *b);

/// Compute the eigenvalues of a pixel's Z matrix.
void compute_eigenvalues(float *hgrad, float *vgrad, int image_height, int image_width, int windowsize, data_wrapper_t *eigenvalues);

/// Calculate the minimum eigenvalue.
float min_eigenvalue(float a, float b, float c, float d);

/// Produce the images horizontal and vertical gradients.
void convolve(float *kernel, float *image, float *resultimage, int image_width, int image_height, int kernel_width, int kernel_height, int half);

/// Creates Gaussian kernel and Gaussian derivative kernel for image gradient/convolution procedure.
void gen_kernel(float *gkernel, float *dkernel, float sigma, int a, int w);

/// Prints out the program help menu.
void help(const char *err);

/// Print out all of the detected features.
void print_features(data_wrapper_t *features, unsigned int count);

#endif