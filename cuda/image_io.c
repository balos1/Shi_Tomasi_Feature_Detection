/*
    File for image reading and writing. Uses stb_image.h for reading image, but currently only supports 1 color channel.
    Author: Cody Balos / Sumedh Naik (now at Intel)
*/


#include "image_io.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define BUFFER 512

#define READ_IMAGE_TEMPLATE(T) { \
    int im_channels; \
    unsigned char *data = stbi_load(name, im_width, im_height, &im_channels, 1); \
	if (data == NULL) { \
		printf("ERROR: Cannot read %s\n", name); \
		exit(0); \
	} \
    *image = (T*)malloc(sizeof(**image) * (*im_width) * (*im_height)); \
    for (int i = 0; i < (*im_width) * (*im_height); ++i) \
        (*image)[i] = (T)data[i]; \
    stbi_image_free(data); \
}

#define WRITE_IMAGE_TEMPLATE(T) { \
	unsigned char *temp_img = (unsigned char *)malloc(sizeof(unsigned char)*im_width*im_height); \
	for(int i = 0;i < (im_width*im_height); i++) \
		temp_img[i] = image[i]; \
	write_imagec(name, temp_img, im_width, im_height); \
	free(temp_img); \
}

void read_image(char *name, double **image, int *im_width, int *im_height) 
{
    READ_IMAGE_TEMPLATE(double)
}

void read_imagef(char *name, float **image, int *im_width, int *im_height) 
{
    READ_IMAGE_TEMPLATE(float)
}

void read_imagei(char *name, int **image, int *im_width, int *im_height) 
{
    READ_IMAGE_TEMPLATE(int)
}

void read_imagec(char *name, unsigned char **image, int *im_width, int *im_height) 
{
    READ_IMAGE_TEMPLATE(unsigned char)
}

void write_image(char *name, double *image, int im_width, int im_height)
{
	WRITE_IMAGE_TEMPLATE(double)
}

void write_imagef(char *name, float *image, int im_width, int im_height)
{
	WRITE_IMAGE_TEMPLATE(float)
}

void write_imagei(char *name, int *image, int im_width, int im_height)
{
	WRITE_IMAGE_TEMPLATE(int)
}

void write_imagec(char *name, unsigned char *image, int im_width, int im_height)
{
	FILE *fop; 
	int im_size=im_width*im_height;
	
	fop=fopen(name,"w+");
	fprintf(fop,"P5\n%d %d\n255\n",im_width,im_height);
	fwrite(image,sizeof(unsigned char),im_size,fop);
	
	fclose(fop);
}