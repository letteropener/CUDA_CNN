#include <cuda.h>
#include <cuda_runtime.h>
#include "others.cu"
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <string>


void genData(float *data, int dim)
{
	for (int i = 0; i < dim*dim; ++i)
	{
		data[i] = (((float)rand() / (float)RAND_MAX) - 0.5f) * 10;
	}
}


float GetElement(Matrix2* mat, int row, int col)
		{
			if (row >= *mat->height || col >= *mat->width) return 0;
			return mat->elements[row * (*mat->width) + col];
		}


void printMatrix(Matrix3* mat){
	//printf("height: %d\n",*mat->height);
	//printf("width: %d\n",*mat->width);
	for (int k = 0; k < *mat->depth; ++k)
			{
				for (int i = 0; i < *mat->height; ++i)
				{
					for (int j = 0; j < *mat->width; ++j)
					{
						printf("%f ", GetElement(&mat->mats[k], i, j));
					}
					printf("\n");
				}

}}


int main(int argc, char* argv[]){

	const int HEIGHT = 28*8;
	const int WIDTH = 28*8;
	const int DEPTH = 3;
	const int FILTER_WIDTH = 3;
	const int FILTER_HEIGHT = 3;
//initial input Matrix in host
	Matrix3 input;
	input.depth = (int*)malloc(sizeof(int));
	input.height = (int*)malloc(sizeof(int));
	input.width = (int*)malloc(sizeof(int));
	*input.depth = DEPTH;
	*input.width = WIDTH;
	*input.height = HEIGHT;
	input.elements = (float *)malloc(sizeof(float)*(HEIGHT * WIDTH * DEPTH));
	input.mats = (Matrix2*)malloc(sizeof(Matrix2)*(DEPTH));
	for (int i = 0; i < DEPTH; ++i)
	{
		input.mats[i].width = (int*)malloc(sizeof(int));
		input.mats[i].height = (int*)malloc(sizeof(int));
		*input.mats[i].height = HEIGHT;
		*input.mats[i].width = WIDTH;
	}
	//printf("Input Matrix\n");
	
	for	(int i = 0; i < DEPTH; ++i){
		input.mats[i].elements = (float*)malloc(sizeof(float)*HEIGHT * WIDTH);
		genData(input.mats[i].elements, HEIGHT);
	}
	//printMatrix(&input);



//initial filter Matrix in host
	Matrix3 filter;
	filter.depth = (int*)malloc(sizeof(int));
	filter.height = (int*)malloc(sizeof(int));
	filter.width = (int*)malloc(sizeof(int));
	*filter.depth = DEPTH;
	*filter.width = FILTER_WIDTH;
	*filter.height = FILTER_HEIGHT;
	filter.elements = (float *)malloc(sizeof(float)*(HEIGHT * WIDTH * DEPTH));
	filter.mats = (Matrix2*)malloc(sizeof(Matrix2)*DEPTH);
	for (int i = 0; i < DEPTH; ++i)
	{
		filter.mats[i].width = (int*)malloc(sizeof(int));
		filter.mats[i].height = (int*)malloc(sizeof(int));
		*filter.mats[i].height = FILTER_HEIGHT;
		*filter.mats[i].width = FILTER_WIDTH;
	}
	//printf("Filter Matrix\n");
	
	
	for	(int i = 0; i < DEPTH; ++i){
		filter.mats[i].elements = (float*)malloc(sizeof(float)*FILTER_WIDTH * FILTER_HEIGHT);
		genData(filter.mats[i].elements, FILTER_HEIGHT);
	}
	
	//printMatrix(&filter);

/*	//Starting timing
	float gpu_elapsed_time_ms;	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
*/
	Convolve(&input, &filter, 1, 1);

/*	cudaEventRecord(stop, 0);	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	printf("Others Time elapsed on GPU %f ms.\n",gpu_elapsed_time_ms);
*/
	




}
