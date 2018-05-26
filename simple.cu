#include <cuda.h>
#include <cuda_runtime.h>
#include "simple.h"
#include <string>
#include <stdio.h>


__device__ float GetElementCuda(Matrix2* mat, int row, int col)
		{
			if (row >= *mat->height || col >= *mat->width) return -999999999;
			return mat->elements[row * (*mat->width) + col];
		}


__global__ void ConvolveKernel(Matrix3* input, Matrix3* output, Matrix3* filters, int filterCount, int stride,int blockSize){

			int gridIdy = blockIdx.y;
			int gridIdx = blockIdx.x;

			int blockRow = threadIdx.y;
			int blockCol = threadIdx.x;
			
			int height = gridIdy * blockSize + blockRow;
			int width = gridIdx * blockSize + blockCol;
			
			//int idx = gridIdx * blockSize * blockSize + gridIdy * blockDim.x * blockSize * blockSize + blockCol + blockRow * blockSize;
			
		if (height >= *input->height - *filters->height + 1 || width >= *input->width - *filters->width + 1) return;
			
			float sum = 0;				
			for (int k = 0; k < *filters->depth; ++k)
			{
				for (int i = 0; i < *filters->height; ++i)
				{
					for (int j = 0; j < *filters->width; ++j)
					{
						//printf("%d,%d,%d\n",k,i,j);
						sum += GetElementCuda(&input->mats[k], height + i, width + j) * GetElementCuda(&filters->mats[k], i, j);
					}
				}
			}
			output->mats[0].elements[(*output->width) * height + width] = sum;
			//printf("%f \n",sum); 
			

}



__global__ void MaxPoolKernel(Matrix3* input, Matrix3* output, int stride, int blockSize)
		{
			
			int gridIdy = blockIdx.y;
			int gridIdx = blockIdx.x;

			int blockRow = threadIdx.y;
			int blockCol = threadIdx.x;
			
			int height = gridIdy * blockSize + blockRow;
			int width = gridIdx * blockSize + blockCol;
			
			//printf("height: %d\n", *output->width);
			if (height >= *output->height || width >= *output->width) return;
			
			float value = -999999999;
			for (int i = 0; i < stride; ++i)
			{
				for (int j = 0; j < stride; ++j)
				{
					float cur_value = GetElementCuda(&input->mats[0], stride*height+i, stride*width+j);
					if (cur_value > value)
					{
						value = cur_value;
					}
				}
			}
			//printf("Setting %d %d of %d to %f\n", xIdx, yIdx, matIdx, value);
			output->mats[0].elements[(*output->width)*height + width] = value;
			//printf("MaxPooling: inside GPU\n");
			//printf("%f\n",value);
		}


__global__ void ConvolveKernel_SharedMemory(Matrix3* input, Matrix3* output, Matrix3* filters, int filterCount, int stride,int blockSize){

			int gridIdy = blockIdx.y;
			int gridIdx = blockIdx.x;

			int blockRow = threadIdx.y;
			int blockCol = threadIdx.x;
			
			int height = gridIdy * blockSize + blockRow;
			int width = gridIdx * blockSize + blockCol;

			__shared__ float filter_shared[3][3][3];
			if(blockRow<3&&blockCol<9){
				filter_shared[blockRow][blockCol/3][blockCol%3]=GetElementCuda(&filters->mats[blockRow],blockCol/3,blockCol%3);			
			}
			
			__syncthreads();
			
			
			//int idx = gridIdx * blockSize * blockSize + gridIdy * blockDim.x * blockSize * blockSize + blockCol + blockRow * blockSize;
			
		if (height >= *input->height - *filters->height + 1 || width >= *input->width - *filters->width + 1) return;
			
			float sum = 0;				
			for (int k = 0; k < *filters->depth; ++k)
			{
				for (int i = 0; i < *filters->height; ++i)
				{
					for (int j = 0; j < *filters->width; ++j)
					{
						//printf("%d,%d,%d\n",k,i,j);
						sum += GetElementCuda(&input->mats[k], height + i, width + j) * filter_shared[k][i][j];
					}
				}
			}
			output->mats[0].elements[(*output->width) * height + width] = sum;
			//printf("%f \n",sum); 
			

}


__global__ void ConvolveKernel_SharedMemory_unroll(Matrix3* input, Matrix3* output, Matrix3* filters, int filterCount, int stride,int blockSize){

			int gridIdy = blockIdx.y;
			int gridIdx = blockIdx.x;

			int blockRow = threadIdx.y;
			int blockCol = threadIdx.x;
			
			int height = gridIdy * blockSize + blockRow;
			int width = gridIdx * blockSize + blockCol;

			__shared__ float filter_shared[3][3][3];
			if(blockRow<3&&blockCol<9){
				filter_shared[blockRow][blockCol/3][blockCol%3]=GetElementCuda(&filters->mats[blockRow],blockCol/3,blockCol%3);			
			}
			
			__syncthreads();
			
			
			//int idx = gridIdx * blockSize * blockSize + gridIdy * blockDim.x * blockSize * blockSize + blockCol + blockRow * blockSize;
			
		if (height >= *input->height - *filters->height + 1 || width >= *input->width - *filters->width + 1) return;
			
			float sum = 0;				
			for (int k = 0; k < *filters->depth; ++k)
			{
				for (int i = 0; i < *filters->height; ++i)
				{
					sum += GetElementCuda(&input->mats[k], height + i, width ) * filter_shared[k][i][0];
					sum += GetElementCuda(&input->mats[k], height + i, width + 1) * filter_shared[k][i][1];
					sum += GetElementCuda(&input->mats[k], height + i, width + 2) * filter_shared[k][i][2];
				}
			}
			output->mats[0].elements[(*output->width) * height + width] = sum;
			//printf("%f \n",sum); 
			

}



/*
__global__ void test(Matrix3* input){
		int idx=threadIdx.x;
		
		if(idx>=1) return;
		
		printf("index = %d,%f\n",idx,input->mats[2].elements[0]); 
}*/
