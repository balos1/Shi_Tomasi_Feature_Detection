/*
	File: serialprogram.c
    Author(s):
		Yang Liu - University of the Pacific, ECPE 293, Spring 2017
		Cody Balos - University of the Pacific, ECPE 293, Spring 2017
	Description:
    	This program implements Shi Tomasi Feature Detection.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include "image_io.h"
#include "stfd.h"
#include "timing.h"

#if !BENCHMARKMODE
#undef TIME_BLOCK_EXEC
#define TIME_BLOCK_EXEC(msg, ...) do {   \
    __VA_ARGS__                          \
} while(0);
#endif

int main(int argc, char **argv)
{
	TIME_BLOCK_EXEC("end_to_end",
	{ // START "END-TO-END" TIMING BLOCK
		// User provided arguments

		// path to the image to process
		char *filepath = NULL;
		// how much information should the program print to the console
		int verbose_lvl = 0;
		// sigma of the gaussian distribution
		float sigma = 1.1;
		// size of a pixel 'neighborhood'
		int windowsize = 4;
		// # of features
		int max_features = 1024;

		// argument parsing logic
		if (argc > 1) {
			if (!strcmp(argv[1], "-h")) {
				help(NULL);
			}
			else if (!strcmp(argv[1], "-v")) {
				verbose_lvl = 1;
				filepath = argv[2];
				if (argc >= 4)
					sigma = atof(argv[3]);
				if (argc >= 5)
					windowsize = atof(argv[4]);
				if (argc >= 6)
					max_features = atoi(argv[5]);
			}
			else if (!strcmp(argv[1], "-vv")) {
				verbose_lvl = 2;
				filepath = argv[2];
				if (argc >= 4)
					sigma = atof(argv[3]);
				if (argc >= 5)
					windowsize = atof(argv[4]);
				if (argc >= 6)
					max_features = atoi(argv[5]);
			}
			else {
				filepath = argv[1];
				if (argc >= 3)
					sigma = atof(argv[2]);
				if (argc >= 4)
					windowsize = atof(argv[3]);
				if (argc >= 5)
					max_features = atoi(argv[4]);
			}
		} else {
			help("You must provide the path to the image to process.");
		}

#if BENCHMARKMODE
		verbose_lvl = 0;
#endif

		if (verbose_lvl > 0) {
			printf("detecting features for %s\n", filepath);
			printf("sigma = %0.3f, windowsize = %d, max_features = %d\n", sigma, windowsize, max_features);
			printf("max threads = %d\n", omp_get_max_threads());
		}

#if BENCHMARKMODE
		printf("detecting features for %s\n", filepath);
		printf("sigma = %0.3f, windowsize = %d, max_features = %d\n", sigma, windowsize, max_features);
		printf("max threads = %d\n", omp_get_max_threads());
#endif

		int width;
		int height;
		int kernel_width;
		int a;

		// calculate kernel width based on sigma
		a = (int)round(2.5 * sigma -.5);
		kernel_width = 2 * a + 1;

		// malloc and read the image to be processed
		float *original_image;
		TIME_BLOCK_EXEC("disk_IO_read",
		{
			read_imagef(filepath, &original_image, &width, &height);
		})

#if BENCHMARKMODE
		printf("image_size\n%dpx\n", width);
#endif

		// malloc and generate the kernels
		float *gkernel = (float *)malloc(sizeof(float) * kernel_width);
		float *dkernel = (float *)malloc(sizeof(float) * kernel_width);
		gen_kernel(gkernel, dkernel, sigma, a, kernel_width);

		// create hgrad and vgrad and temp
		float *hgrad = (float *)malloc(sizeof(float) * width * height);
		float *vgrad =  (float *)malloc(sizeof(float) * width * height);
		float *tmp_image = (float *)malloc(sizeof(float) * width * height);

		// convolve to get the vgrad and hgrad
		TIME_BLOCK_EXEC("convolution",
		{
			convolve(gkernel, original_image, tmp_image, width, height, kernel_width, 1, a);
			convolve(dkernel, tmp_image, vgrad, width, height, 1, kernel_width, a);
			convolve(gkernel, original_image,tmp_image, width, height, 1, kernel_width, a);
			convolve(dkernel, tmp_image, hgrad, width, height, kernel_width, 1, a);
		})
		free(tmp_image);
		free(gkernel);
		free(dkernel);

		// Compute the eigenvalues of each pixel's z matrix. After this we can free the gradients.
		data_wrapper_t *eigenvalues = (data_wrapper_t *)malloc(sizeof(data_wrapper_t) * width * height);
		TIME_BLOCK_EXEC("compute_eigenvalues",
		{
			compute_eigenvalues(hgrad, vgrad, height, width, windowsize, eigenvalues);
		})
		free(hgrad);
		free(vgrad);

		// Find the features based on the eigenvalues.
		data_wrapper_t *features;
		unsigned int features_count;
		TIME_BLOCK_EXEC("find_features",
		{
			features_count = find_features(eigenvalues, max_features, width, height, &features);
		})
		free(eigenvalues);

		if (verbose_lvl > 0) {
			printf("%d features detected\n", features_count);
		}
		if (verbose_lvl > 1) {
			printf("\t");
			print_features(features, features_count);
		}

		// Mark the features in the output image.
		TIME_BLOCK_EXEC("draw_features",
		{
			draw_features(features, features_count, original_image, width, height);
		})
		free(features);

		// Now we write the output.
		char corner_image[30];
		sprintf(corner_image, "corners.pgm");
		TIME_BLOCK_EXEC("disk_IO_write",
		{
			write_imagef(corner_image, original_image, width, height);
		})

		// Free stuff leftover.
		free(original_image);
	}) // END "END-TO-END" TIMING BLOCK

	return 0;
}

void draw_features(data_wrapper_t *features, unsigned int count, float *image, int image_width, int image_height)
{
	int radius = image_width*0.0025;
	for (int i = 0; i < count; ++i) {
		int x = features[i].x;
		int y = features[i].y;
		for (int  j = -1 * radius; j <= radius; j++) {
			for (int k = -1 * radius; k <= radius; k++) {
				if ((x+j) >= 0 && (x+j) < image_height && (y+k) >= 0 && (y+k) < image_width)
					image[(x+j) * image_width + (y+k) ] = 0;
			}
		}
	}
}

unsigned int find_features(data_wrapper_t *eigenvalues, int max_features, int image_width, int image_height, data_wrapper_t **features)
{
	size_t image_size = image_height*image_width;

	TIME_BLOCK_EXEC("features_sort",
	{
		// Sort eigenvalues in descending order while keeping their corresponding pixel index in the image.
		qsort(eigenvalues, image_height*image_width, sizeof *eigenvalues, sort_data_wrapper_value_desc);
	})

	// Create the features buffer based on the max_features value (acts as a percentage of the image size).
	*features = (data_wrapper_t*)malloc(sizeof(data_wrapper_t)*max_features);

	// Fill the features buffer!
	unsigned int features_count = 0;
	const int ignore_x = 3; // ignore this many pixels rows from top/bottom of image
	const int ignore_y = 3; // ignore this many pixels columns from left/right of image
	for (int i = 0; i < image_size && features_count < max_features; ++i) {
		// Ignore top left, top right, bottom right, bottom left edges of image.
		if (eigenvalues[i].x <= ignore_x || eigenvalues[i].y <= ignore_y ||
    		eigenvalues[i].x >= image_height-1-ignore_x || eigenvalues[i].y >= image_width-1-ignore_y) {
			continue;
		}

		// Have to seed the first feature so we have a place to start.
		if (features_count == 0) {
			(*features)[0] = eigenvalues[i];
			features_count++;
		}

		// Check if prospective feature is more than 8 manhattan distance away from any existing feature.
        int is_good = 1;
		for (int j = 0; j < features_count; ++j) {
			int manhattan = abs((*features)[j].x - eigenvalues[i].x) + abs((*features)[j].y - eigenvalues[i].y);
			if (manhattan <= 8) {
				is_good = 0;
				break;
			}
		}

        // If the prospective feature was at least 8 manhattan distance from all existing features, then we can add it.
		if (is_good) {
			(*features)[features_count] = eigenvalues[i];
			features_count++;
		}
	}

	return features_count;
}

int sort_data_wrapper_value_desc(const void *a, const void *b)
{
	const data_wrapper_t *aa = (const data_wrapper_t *) a;
	const data_wrapper_t *bb = (const data_wrapper_t *) b;
	return (aa->data < bb->data) - (aa->data > bb->data);
}

int sort_data_wrapper_index_asc(const void *a, const void *b)
{
	const data_wrapper_t *aa = (const data_wrapper_t *) a;
	const data_wrapper_t *bb = (const data_wrapper_t *) b;

	if (aa->x == bb->x)
		return ((aa->y > bb->y) - (aa->y < bb->y));
	else
		return ((aa->x > bb->x) - (aa->x < bb->x));
}

void compute_eigenvalues(float *hgrad, float *vgrad, int image_height, int image_width, int windowsize, data_wrapper_t *eigenvalues)
{
	int w = floor(windowsize/2);
 	int i, j;

	#pragma omp parallel for private(j)
	for (i = 0; i < image_height; i++) {
		for (j = 0; j < image_width; j++) {
			float ixx_sum = 0;
			float iyy_sum = 0;
			float ixiy_sum = 0;

			for (int k = 0; k < windowsize; k++) {
				for (int m = 0; m < windowsize; m++) {
					int offseti = -1 * w + k;
					int offsetj = -1 * w + m;
					if (i+offseti >= 0 && i+offseti < image_height && j + offsetj >= 0 && j+offsetj < image_width){
						ixx_sum += hgrad[(i +offseti) * image_width  + (j + offsetj)] * hgrad[(i +offseti) * image_width  + (j + offsetj)];
						iyy_sum += vgrad[(i +offseti) * image_width  + (j + offsetj)] * vgrad[(i +offseti) * image_width  + (j + offsetj)];
						ixiy_sum += hgrad[(i +offseti) * image_width  + (j + offsetj)] * vgrad[(i +offseti) * image_width  + (j + offsetj)];
					}
				}
			}

			eigenvalues[i*image_width+j].x = i;
			eigenvalues[i*image_width+j].y = j;
			eigenvalues[i*image_width+j].data = min_eigenvalue(ixx_sum, ixiy_sum, ixiy_sum, iyy_sum);
		}
	}
}

float min_eigenvalue(float a, float b, float c, float d)
{
	float ev_one = (a + d)/2 + pow(((a + d) * (a + d))/4 - (a * d - b * c), 0.5);
	float ev_two = (a + d)/2 - pow(((a + d) * (a + d))/4 - (a * d - b * c), 0.5);
	if (ev_one >= ev_two){
		return ev_two;
	}
	else{
		return ev_one;
	}
}

void convolve(float *kernel, float *image, float *resultimage, int image_width, int image_height, int kernel_width, int kernel_height, int half)
{
	int i, j;

	#pragma omp parallel for private(j)
	for (i = 0; i < image_height; i++) {
		for (j = 0; j < image_width; j++) {
			// reset accumulator when "focused" pixel changes
			float sum = 0.0;
			// for each item in the kernel
			for (int k = 0; k < kernel_height; k++) {
				for (int m = 0; m < kernel_width; m++) {
					int offseti = -1 * (kernel_height/2) + k;
					int offsetj = -1 * (kernel_width/2) + m;
					// Check to make sure we are in the bounds of the image.
					if (i+offseti >= 0 && i+offseti < image_height && j + offsetj >= 0 && j+offsetj < image_width)
						sum+=(float)(image[(i+offseti) * image_width + (j+offsetj)])*kernel[k*kernel_width +m];
				}
			}
			resultimage[i * image_width + j] = sum;
		}
	}
}

void gen_kernel(float *gkernel, float *dkernel, float sigma, int a, int w)
{
	int i;
	float sum_gkern = 0;
	float sum_dkern = 0;

	for (i = 0; i < w; i++) {
		gkernel[i] = (float)exp( (float)(-1.0 * (i-a) * (i-a)) / (2 * sigma * sigma));
		dkernel[i] = (float)(-1 * (i - a)) * (float)exp( (float)(-1.0 * (i-a) * (i-a)) / (2 * sigma * sigma));
		sum_gkern = sum_gkern + gkernel[i];
		sum_dkern = sum_dkern - (float)i * dkernel[i];
	}

	//reverse the kernel by creating a new kernel, yes not ideal
	float *newkernel = (float *)malloc(sizeof(float) * w);
	for (i = 0; i < w; i++) {
		dkernel[i] = dkernel[i] / sum_dkern;
		gkernel[i] = gkernel[i] / sum_gkern;
		newkernel[w-i] = dkernel[i];
	}

	//copy new kernel back in
	for (i = 0; i < w; i++)
		dkernel[i] = newkernel[i+1];

	free(newkernel);
}

void help(const char *err)
{
    if (err != NULL)
        printf("%s\n", err);
    printf("usage: ./stfd [-v,-vv] <full path to the image> [sigma] [windowsize] [num_features] \n");
    printf("flags:\n");
    printf("\t-h: show this help menu\n");
    printf("\t-v: output basic execution information\n");
    printf("\t-vv: output all information... good for debugging\n");
	printf("arguments:\n");
	printf("\tsigma: the sigma value for the Gaussian distribution used to form the convolution mask.\n");
	printf("\twindowsize: the size of a pixel 'neighborhood' in an image\n");
	printf("\tnum_features: how many features to extract\n");
    exit(0);
}

void print_features(data_wrapper_t *features, unsigned int count)
{
	// Sort the features by
	qsort(features, count, sizeof *features, sort_data_wrapper_index_asc);
	for (unsigned int i = 0; i < count; ++i) {
		if (i % 15 != 0 || i == 0)
			printf("(%d,%d) ", features[i].x, features[i].y);
		else
			printf("(%d,%d)\n\t", features[i].x, features[i].y);
	}
	printf("\n");
}
