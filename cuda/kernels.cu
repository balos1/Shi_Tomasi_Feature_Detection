/*
	File: kernels.cu
    Author(s):
		Cody Balos - University of the Pacific, ECPE 293, Spring 2017
	Description:
    	GPU kernels and GPU kernel wrappers for CUDA Shi Tomasi Feature Detection program.
*/

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "timing.h"
#include "config.h"
#include "kernels.h"

#if BENCHMARKMODE
#define TIME_BLOCK_EXEC(msg, ...) do {                                           \
    struct timeval __start, __end;                                               \
    gettimeofday(&__start, NULL);                                                \
    cudaDeviceSynchronize();                                                     \
    __VA_ARGS__                                                                  \
    cudaDeviceSynchronize();                                                     \
    gettimeofday(&__end, NULL);                                                  \
    printf("%s\n%ldms\n", msg, get_elapsed_time(__start, __end));                \
} while(0);
#else
#define TIME_BLOCK_EXEC(msg, ...) do {   \
    __VA_ARGS__                          \
} while(0);
#endif

// CPU host buffers to store convolution masks.
// Fixing the width of the mask so we can leverage constant memory.
__constant__ float d_gaussian_kernel[MAX_KERNEL_WIDTH];
__constant__ float d_gaussian_deriv[MAX_KERNEL_WIDTH];

/// A GPU kernel for convolution with the gaussian kernel. Input image can be either int or float BUT the output
/// is always float.

__device__ long d_flops;
__device__ long d_gma;

__global__
void convolve_gaussian(float *in_image, int width, int height, int mask_width, int mask_height, float *out_image)
{
    // Each thread will load its respective element into the shared memory for the thread block.
    // This leverages the GPU's L2 Cache, providing significant speedup.
    extern __shared__ float in_shared[];

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    in_shared[threadIdx.x*blockDim.x + threadIdx.y] = in_image[i*width + j];

    // Sync all threads after collobaritve loading prior to moving on with convolution.
    __syncthreads();

    // Now perform the actual convolution operation.
    if (i < height && j < width)
    {
        float sum = 0;
        for(int k = 0; k < mask_height; k++) {
            for(int m = 0; m < mask_width; m++) {
                int offseti = k-(mask_height/2);
                int offsetj = m-(mask_width/2);
                // If the element is in shared memory for the block, then load from it. Otherwise we have to fo to
                // GPU main memory to get it.
                if((offseti+threadIdx.x) >= 0 && (offseti+threadIdx.x) < blockDim.x && (offsetj+threadIdx.y) >= 0 && (offsetj+threadIdx.y) < blockDim.y) {
                    sum += (float)(in_shared[(offseti+threadIdx.x)*blockDim.x+(offsetj+threadIdx.y)])*d_gaussian_kernel[k*mask_width+m];
                    #if BENCHMARKMODE
                    d_flops += 2;
                    #endif
                }
                else if (offseti+i >= 0 && offseti+i < height && offsetj+j >= 0 && offsetj+j < width) {
                    sum += (float)(in_image[(offseti+i)*width+(offsetj+j)])*d_gaussian_kernel[k*mask_width+m];
                    #if BENCHMARKMODE
                    d_flops += 2;
                    d_gma++;
                    #endif
                }
            }
        }
        out_image[i*width+j] = (float)sum;
    }
}

/// A GPU kernel for convolution with the gaussian derivative kernel. Input image can be either int or float BUT the
/// output is always float.
__global__
void convolve_dgaussian(float *in_image, int width, int height, int mask_width, int mask_height, float *out_image)
{
    // Each thread will load its respective element into the shared memory for the thread block.
    // This leverages the GPU's L2 Cache, providing significant speedup.
    extern __shared__ float in_shared[];

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    in_shared[threadIdx.x*blockDim.x + threadIdx.y] = in_image[i*width + j];

    // Sync all threads after collobaritve loading prior to moving on with convolution.
    __syncthreads();

    // Now perform the actual convolution operation.
    if (i < height && j < width)
    {
        float sum = 0;
        for(int k = 0; k < mask_height; k++) {
            for(int m = 0; m < mask_width; m++) {
                int offseti = k-(mask_height/2);
                int offsetj = m-(mask_width/2);
                // If the element is in shared memory for the block, then load from it. Otherwise we have to go to
                // GPU global memory to get it.
                if ((offseti+threadIdx.x) >= 0 && (offseti+threadIdx.x) < blockDim.x && (offsetj+threadIdx.y) >= 0 && (offsetj+threadIdx.y) < blockDim.y) {
                    sum += (float)(in_shared[(offseti+threadIdx.x)*blockDim.x+(offsetj+threadIdx.y)])*d_gaussian_deriv[k*mask_width+m];
                    #if BENCHMARKMODE
                    d_flops += 2;
                    #endif
                }
                else if (offseti+i >= 0 && offseti+i < height && offsetj+j >= 0 && offsetj+j < width) {
                    sum += (float)(in_image[(offseti+i)*width+(offsetj+j)])*d_gaussian_deriv[k*mask_width+m];
                    #if BENCHMARKMODE
                    d_flops += 2;
                    d_gma++;
                    #endif
                }
            }
        }
        out_image[i*width+j] = (float)sum;
    }
}

/// It calculates the eigenvalues for the given matrix and returns the lesser one.
__device__
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

/// A GPU kernel which computes the eigenvalues of the Z matrix for every pixel.
__global__
void compute_eigenvalues(float *hgrad, float *vgrad, int height, int width, int windowsize, data_wrapper_t *eigenvalues)
{
    // Leverage the shared L2 cache.
    extern __shared__ float shared[];
    float *hgrad_shared = (float*)&shared[0];
    float *vgrad_shared = (float*)&shared[blockDim.x*blockDim.y];

 	int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    // Perform collaborative loading of shared memory.
    hgrad_shared[threadIdx.x*blockDim.x+threadIdx.y] = hgrad[i*width+j];
    vgrad_shared[threadIdx.x*blockDim.x+threadIdx.y] = vgrad[i*width+j];

    // Must sync all threads prior to computing eigenvalues because shared memory must be completely populated first.
    __syncthreads();

    int w = floor((float) windowsize/2);
    if (i < height && j < width) {
        float ixx_sum = 0, iyy_sum = 0, ixiy_sum = 0;
        for (int k = 0; k < windowsize; k++) {
            for (int m = 0; m < windowsize; m++) {
                int offseti = - w + k;
                int offsetj = - w + m;
                // If the element is in shared memory for the block, then load from it. Otherwise we have to go to
                // GPU global memory to get it.
                if ((threadIdx.x+offseti) >= 0 && (threadIdx.x+offseti) < blockDim.x && (threadIdx.y+offsetj) >= 0 && (threadIdx.y+offsetj) < blockDim.y) {
                    float h_val = hgrad_shared[(threadIdx.x+offseti)*blockDim.x+(threadIdx.y+offsetj)];
                    float v_val = vgrad_shared[(threadIdx.x+offseti)*blockDim.x+(threadIdx.y+offsetj)];
                    ixx_sum += h_val*h_val;
                    iyy_sum += v_val*v_val;
                    ixiy_sum += h_val*v_val;
                    #if BENCHMARKMODE
                    d_flops += 6;
                    #endif
                } else if ((i+offseti) >= 0 && (i+offseti) < height && (j+offsetj) >= 0 && (j+offsetj) < width) {
                    float h_val = hgrad[(i+offseti)*width+(j+offsetj)];
                    float v_val = vgrad[(i+offseti)*width+(j+offsetj)];
                    ixx_sum += h_val*h_val;
                    iyy_sum += v_val*v_val;
                    ixiy_sum += h_val*v_val;
                    #if BENCHMARKMODE
                    d_flops += 6;
                    d_gma += 2;
                    #endif
                }
            }
        }
        eigenvalues[i*width+j].x = i;
        eigenvalues[i*width+j].y = j;
        eigenvalues[i*width+j].data = min_eigenvalue(ixx_sum, ixiy_sum, ixiy_sum, iyy_sum);
    }
}

/// Comparator for sorting data_wrapper_t array in descending order of value.
struct sort_data_wrapper_value_desc {
    __host__ __device__
    bool operator()(const data_wrapper_t& a, const data_wrapper_t& b)
    {
        return a.data > b.data;
    }
};

void cuda_compute_eigenvalues(float *d_hgrad, float *d_vgrad, int height, int width, int windowsize, data_wrapper_t **d_eigenvalues)
{
    // Configure CUDA grids and blocks
	dim3 dimGrid(ceil(height/BLOCKSIZE_X), ceil(width/BLOCKSIZE_Y), 1);
    dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, 1);

    // Allocate GPU device buffers
    gpu_err_chk( cudaMalloc((void **)d_eigenvalues, sizeof(**d_eigenvalues)*width*height) );

    TIME_BLOCK_EXEC("compute_eigenvalues_kernel",
    {
        typeof(d_flops) flops;
        typeof(d_gma) gma;

        // Execute GPU kernel
        compute_eigenvalues<<<dimGrid,dimBlock,2*sizeof(*d_hgrad)*dimBlock.x*dimBlock.y>>>(
            d_hgrad,
            d_vgrad,
            height,
            width,
            windowsize,
            *d_eigenvalues
        );

        #if DEBUG
        gpu_err_chk( cudaPeekAtLastError() );
        gpu_err_chk( cudaDeviceSynchronize() );
        #endif

        gpu_err_chk( cudaMemcpyFromSymbol(&flops, d_flops, sizeof(flops), 0, cudaMemcpyDeviceToHost) );
        gpu_err_chk( cudaMemcpyFromSymbol(&gma, d_gma, sizeof(gma), 0, cudaMemcpyDeviceToHost) );

        #if BENCHMARKMODE
        printf("FLOPS\n%ld\n", flops);
        printf("GMA\n%ld\n", gma);
        #endif
    })

    // Free device memory items which wont be used again.
    gpu_err_chk( cudaFree(d_hgrad) );
    gpu_err_chk( cudaFree(d_vgrad) );
}

void cuda_convolve(float *image, int width, int height, float *gk, float *gkd, int k_width, int k_height, float **d_hgrad,
				   float **d_vgrad)
{
	// Configure CUDA grids and blocks
	dim3 dimGrid(ceil(height/BLOCKSIZE_X), ceil(width/BLOCKSIZE_Y), 1);
    dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, 1);

	// GPU device buffer for original image.
    float *d_image;

    // GPU buffers to hold intermediate convolution results.
    float *d_temp;

    // Allocate all of the device memory needed.
    gpu_err_chk( cudaMalloc((void **)&d_image, sizeof(float)*width*height) );
    gpu_err_chk( cudaMalloc((void **)&d_temp, sizeof(float)*width*height) );
    gpu_err_chk( cudaMalloc((void **)d_hgrad, sizeof(float)*width*height) );
    gpu_err_chk( cudaMalloc((void **)d_vgrad, sizeof(float)*width*height) );

    TIME_BLOCK_EXEC("host_to_device",
    {
        // Offload all of the data to GPU device for convolution.
        gpu_err_chk( cudaMemcpy(d_image, image, sizeof(float)*width*height, cudaMemcpyHostToDevice) );

        // Also need to offload constant memory.
        gpu_err_chk( cudaMemcpyToSymbol(d_gaussian_kernel, gk, sizeof(float)*MAX_KERNEL_WIDTH, 0, cudaMemcpyHostToDevice) );
        gpu_err_chk( cudaMemcpyToSymbol(d_gaussian_deriv, gkd, sizeof(float)*MAX_KERNEL_WIDTH, 0, cudaMemcpyHostToDevice) );
    })

	const size_t SHARED_MEM_SIZE = sizeof(float)*dimBlock.x*dimBlock.y;
    TIME_BLOCK_EXEC("convolution_kernel",
    {
        // Horizontal gradient. Uses verical kernel then horizontal derivative.
        convolve_gaussian<<<dimGrid, dimBlock, SHARED_MEM_SIZE>>>(
            d_image,
            width,
            height,
            1,
            k_width,
            d_temp
        );

        #if DEBUG
        gpu_err_chk( cudaPeekAtLastError() );
        gpu_err_chk( cudaDeviceSynchronize() );
        #endif

        convolve_dgaussian<<<dimGrid, dimBlock, SHARED_MEM_SIZE>>>(
            d_temp,
            width,
            height,
            k_width,
            1,
            *d_hgrad
        );

        #if DEBUG
        gpu_err_chk( cudaPeekAtLastError() );
        gpu_err_chk( cudaDeviceSynchronize() );
        #endif

        // Vertical gradient. Uses horizontal kernel then vertical derivative.
        convolve_gaussian<<<dimGrid, dimBlock, SHARED_MEM_SIZE>>>(
            d_image,
            width,
            height,
            k_width,
            1,
            d_temp
        );

        #if DEBUG
        gpu_err_chk( cudaPeekAtLastError() );
        gpu_err_chk( cudaDeviceSynchronize() );
        #endif

        convolve_dgaussian<<<dimGrid, dimBlock, SHARED_MEM_SIZE>>>(
            d_temp,
            width,
            height,
            1,
            k_width,
            *d_vgrad
        );

        #if DEBUG
        gpu_err_chk( cudaPeekAtLastError() );
        gpu_err_chk( cudaDeviceSynchronize() );
        #endif
    }) // END CONVOLUTION_KERNEL TIMe BLOCK

    // Free device memory items that wont be used again.
	gpu_err_chk( cudaFree(d_temp) );
    gpu_err_chk( cudaFree(d_image) );
}

unsigned int cuda_find_features(data_wrapper_t *d_eigenvalues, int max_features, int image_width, int image_height,
                                data_wrapper_t **features)
{
    size_t image_size = image_height*image_width;

	// Sort eigenvalues in descending order while keeping their corresponding pixel index in the image.
    // The sorting is done on the GPU device.
    thrust::device_ptr<data_wrapper_t> thrust_d_eigenvalues(d_eigenvalues);
    TIME_BLOCK_EXEC("find_features_thrust",
    {
	    thrust::sort(thrust_d_eigenvalues, thrust_d_eigenvalues + image_size, sort_data_wrapper_value_desc());
    })
    d_eigenvalues = thrust::raw_pointer_cast(thrust_d_eigenvalues);

    // Host buffer for the eigenvalues.
    data_wrapper_t *eigenvalues = (data_wrapper_t*)malloc(sizeof(data_wrapper_t)*image_size);

    // Copy the sorted eigenvalues back to host memory so we can proceed.
    TIME_BLOCK_EXEC("device_to_host",
    {
        gpu_err_chk( cudaMemcpy(eigenvalues, d_eigenvalues, image_size, cudaMemcpyDeviceToHost) );
    })

    // Free device copy of eigenvalues because it is not needed anymore.
    gpu_err_chk( cudaFree(d_eigenvalues) );

	// Create the features buffer.
	*features = (data_wrapper_t*)malloc(sizeof(data_wrapper_t)*max_features);

    unsigned int features_count = 0;
    TIME_BLOCK_EXEC("CPU_find_features",
    {
        // Fill the features buffer! We do this on CPU because it doesn't take long usually and because it is not highly
        // data parallel.
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
    })

    free(eigenvalues);

    return features_count;
}
