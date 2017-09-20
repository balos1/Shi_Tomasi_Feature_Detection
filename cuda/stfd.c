/*
	File: cudaprogram.c
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
#include "config.h"
#include "image_io.h"
#include "stfd.h"
#include "timing.h"

#if !BENCHMARKMODE
#undef TIME_BLOCK_EXEC
#define TIME_BLOCK_EXEC(msg, ...) do {   \
    __VA_ARGS__                          \
} while(0);
#endif

extern void cuda_convolve(float *image, int width, int height, float *gk, float *gkd, int k_width, int k_height,
						  float **d_hgrad, float **d_vgrad);
extern void cuda_compute_eigenvalues(float *d_hgrad, float *d_vgrad, int height, int width, int windowsize,
									 data_wrapper_t **d_eigenvalues);
unsigned int cuda_find_features(data_wrapper_t *d_eigenvalues, int max_features, int image_width, int image_height,
                                data_wrapper_t **features);

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

		if (verbose_lvl > 0) {
			printf("detecting features for %s\n", filepath);
			printf("sigma = %0.3f, windowsize = %d, max_features = %d\n", sigma, windowsize, max_features);
		}

		int width;
		int height;
		int kernel_width;
		int a;

		// calculate kernel width based on sigma
		a = (int)round(2.5 * sigma -.5);
		kernel_width = 2 * a + 1;

		// Constant memory means we have to define a max kernel width and consequently a max sigma.
		if (kernel_width > MAX_KERNEL_WIDTH) {
			sigma = ((MAX_KERNEL_WIDTH-1)/2 + 0.5)/2.5;
			a = ceil((float)(2.5*sigma-0.5));
			kernel_width = 2*a+1;
			printf("Sigma value chosen is too large, using maximum (%f) instead.\n", sigma);
		}

		// malloc and read the image to be processed
		float *original_image;
		TIME_BLOCK_EXEC("disk_IO_read",
		{
			read_imagef(filepath, &original_image, &width, &height);
		})

#if BENCHMARKMODE
		printf("image_size\n%dpx\n", width);
#endif


		// malloc and generate the convolution kernels/masks
		float *gkernel = (float*)malloc(sizeof(float) * kernel_width);
		float *dkernel = (float*)malloc(sizeof(float) * kernel_width);
		gen_kernel(gkernel, dkernel, sigma, a, kernel_width);

		// GPU pointers for vertical and horizontal gradients
		float *d_hgrad, *d_vgrad;

		// convolve to get the vgrad and hgrad
		cuda_convolve(original_image, width, height, gkernel, dkernel, kernel_width, 1, &d_hgrad, &d_vgrad);
		free(gkernel);
		free(dkernel);

		// Compute the eigenvalues of each pixel's z matrix. After this we can free the gradients.
		data_wrapper_t *d_eigenvalues;
		cuda_compute_eigenvalues(d_hgrad, d_vgrad, height, width, windowsize, &d_eigenvalues);

		// Find the features based on the eigenvalues.
		data_wrapper_t *features;
		unsigned int features_count;
		features_count = cuda_find_features(d_eigenvalues, max_features, width, height, &features);

		if (verbose_lvl)
			printf("%d features detected\n", features_count);
		if (verbose_lvl > 1) {
			printf("\t");
			print_features(features, features_count);
		}

		// Mark the features in the output image.
		TIME_BLOCK_EXEC("CPU_draw_features",
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
		for (int k = -1 * radius; k <= radius; k++ ) {
			for (int m = -1 * radius; m <= radius; m++) {
				if ((x+k) >= 0 && (x+k) < image_height && (y+m) >= 0 && (y+m) <image_width)
					image[(x+k) * image_width + (y+m) ] = 0;
			}
		}
	}
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

void gen_kernel(float *gkernel, float *dkernel, float sigma, int a, int w)
{
	int i;
	float sum_gkern;
	float sum_dkern;
	sum_gkern= 0;
	sum_dkern= 0;
	for(i = 0; i < w; i++){
		gkernel[i] = (float)exp( (float)(-1.0 * (i-a) * (i-a)) / (2 * sigma * sigma));
		dkernel[i] = (float)(-1 * (i - a)) * (float)exp( (float)(-1.0 * (i-a) * (i-a)) / (2 * sigma * sigma));
		sum_gkern = sum_gkern + gkernel[i];
		sum_dkern = sum_dkern - (float)i * dkernel[i];
	}

	//reverse the kernel by creating a new kernel, yes not ideal
	float *newkernel = (float *)malloc(sizeof(float) * w);
	for (i = 0; i < w; i++){
		dkernel[i] = dkernel[i] / sum_dkern;
		gkernel[i] = gkernel[i] / sum_gkern;
		newkernel[w-i] = dkernel[i];
	}

	//copy new kernel back in
	for (i = 0; i < w; i++){
		dkernel[i] = newkernel[i+1];
	}
	free(newkernel);
}

void help(const char *err)
{
    if (err != NULL)
        printf("%s\n", err);
    printf("usage: ./stfd [-v,-vv] <full path to the image> [sigma] [windowsize] [sensitivity] \n");
    printf("flags:\n");
    printf("\t-h: show this help menu\n");
    printf("\t-v: output basic execution information\n");
    printf("\t-vv: output all information... good for debugging\n");
	printf("arguments:\n");
	printf("\tsigma: the sigma value for the Gaussian distribution used to form the convolution mask.\n");
	printf("\twindowsize: the size of a pixel 'neighborhood' in an image\n");
	printf("\tsensitivity: determines the amount of features to detect... can be 0.0 to 1.0\n");
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