#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu.h"
#include <string>

__device__ float GetElementCuda(Matrix2* mat, int row, int col)
		{
			if (row >= mat->height || col >= mat->width) return 0;
			return mat->elements[row * mat->width + col];
		}

__device__ float GetElementCuda(Matrix3* mat, int row, int col, int matIdx)
		{
			if (row >= mat->height || col >= mat->width || matIdx >= mat->depth) return 0;
			return mat->elements[matIdx * mat->height * mat->width + row * mat->width + col];
		}

__device__ void SetElementCuda(Matrix2* mat, int row, int col, int value)
		{
			if (row >= mat->height || col >= mat->width) return;
			mat->elements[row * mat->width + col] = value;
		}

__device__ void SetElementCuda(Matrix3* mat, int row, int col, int value, int matIdx)
		{
			if (row >= mat->height || col >= mat->width) return;
			(mat->elements + matIdx * mat->height * mat->width)[row * mat->width + col] = value;
		}

__global__ void MaxPoolKernel(Matrix3 input, Matrix3 output, int stride, int blockSize)
		{
			int gridIdx = blockIdx.y;
			int matIdx = blockIdx.x;

			int blockRow = threadIdx.y;
			int blockCol = threadIdx.x;

			int idx = gridIdx * blockSize * blockSize + blockRow + blockCol * blockSize;
			if (idx >= output.height * output.width) return;
			//printf("This kernel runs %d of matIdx %d gridIdx %d & maxSize %d\n",
			//	idx, matIdx, gridIdx, output.height * output.width);

			int xIdx = idx % output.height;
			int yIdx = idx / output.height;
			int origXIdx = xIdx * stride;
			int origYIdx = yIdx * stride;
			//printf("map %d %d to %d %d\n",
			//	xIdx, yIdx, origXIdx, origYIdx);

			float value = -999999999;
			for (int i = 0; i < stride; ++i)
			{
				for (int j = 0; j < stride; ++j)
				{
					float cur_value = GetElementCuda(&input, origXIdx+i, origYIdx+j, matIdx);
					if (cur_value > value)
					{
						value = cur_value;
					}
				}
			}
			//printf("Setting %d %d of %d to %f\n", xIdx, yIdx, matIdx, value);
			SetElementCuda(&output, xIdx, yIdx, value, matIdx);
		}

__global__ void ConvolveTransInputKernel(Matrix3 input, float* output,
			int patchWidth, int patchHeight,
			int filterWidth, int filterHeight, int stride, int n)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;
			int patchSize = filterWidth * filterHeight * input.depth;

			int idx = index * patchSize;
			int patchCol = index % patchWidth;
			int patchRow = index / patchWidth;
			int curRow = patchRow * stride;
			int curCol = patchCol * stride;

			//printf("idx %d patch row %d col %d org row %d col %d\n", idx, patchRow, patchCol, curRow, curCol);

			for (int i = 0; i < input.depth; ++i)
			{
				for (int j = 0; j < filterHeight; ++j)
				{
					for (int k = 0; k < filterWidth; ++k)
					{
						output[idx] = GetElementCuda(&input, curRow + j, curCol + k, i);
						idx++;
					}
				}
			}
		}


float* ConvolveTransInput(Matrix3 input, int filterWidth, int filterHeight, int stride)
		{
			// ideally, each row is a basic unit(thread).
			// grid.x is the number of grids(if block not enough for all the rows).
			// each row contains filterSize * depth numbers. 
			int blockSize = 128;
			int height = (input.height - filterHeight) / stride + 1;
			int width = (input.width - filterWidth) / stride + 1;
			int patchCount = height * width;
			int patchSize = filterWidth * filterHeight * input.depth;

			float* resultCuda;
			int matSize = patchCount * patchSize;
			cudaMalloc(&resultCuda, sizeof(float) * matSize);

			dim3 dimGrid((patchCount + blockSize - 1) / blockSize);
			ConvolveTransInputKernel << <dimGrid, blockSize >> >(input, resultCuda, 
				width, height,
				filterWidth, filterHeight, stride, patchCount);
			checkCUDAError("Convolve trans failes.");

			//UTestCheckCudaInputMatT(resultCuda, &input, filterWidth, filterHeight, stride);

			float* resultTransCuda;
			cudaMalloc(&resultTransCuda, sizeof(float) * matSize);

			dim3 fullBlocksPerGrid((matSize + blockSize - 1) / blockSize);
			trans << <fullBlocksPerGrid, blockSize >> >(resultCuda, resultTransCuda,
				patchSize, patchCount);
			cudaFree(resultCuda);

			return resultTransCuda;
		}
