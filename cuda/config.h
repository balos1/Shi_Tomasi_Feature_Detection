/*
	File: config.h
    Author(s):
		Cody Balos - University of the Pacific, ECPE 293, Spring 2017
	Description:
    	Settings for Shi Tomasi Feature Detection Program.
*/

// The maximum size of a convolution kernel.
#define MAX_KERNEL_WIDTH 32

// BLOCKSIZE_X*BLOCKSIZE_Y = number of threads.
#define BLOCKSIZE_X NBLOCKS
#define BLOCKSIZE_Y NBLOCKS

// Turn execution timers on/off.
#define CUDA_TIMERS 1