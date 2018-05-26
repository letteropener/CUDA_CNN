#include <cuda.h>
#include <cuda_runtime.h>
#include "others.h"
#include <string>
#include <stdio.h>

const int blockSizeMat = 16;
__global__ void test(Matrix3* input){
		int idx=threadIdx.x;
		
		if(idx>=1) return;
		printf("asdf\n");
		printf("index = %d,%f\n",idx,input->elements[1]); 
}

__device__ float GetElement(const float *A, int row, int col, int wA)
		{
			return A[row * wA + col];
		}

__device__ void SetElement(float *A, int row, int col, int wA, float value)
		{
			A[row * wA + col] = value;
		}

__device__ float* GetSubMatrix(float *A, int row, int col, int wA)
		{
			return &A[row * wA * blockSizeMat + blockSizeMat * col];
		}

__global__ void matDot(float *A, float *B, float *C, int hA, int wA, int wB)
		{
			int blockRow = blockIdx.y;
			int blockCol = blockIdx.x;

			float *Csub = GetSubMatrix(C, blockRow, blockCol, wB);

			float Cvalue = 0.0;

			int row = threadIdx.y;
			int col = threadIdx.x;

			for (int m = 0; m < (wA + blockSizeMat - 1) / blockSizeMat; ++m)
			{
				float *Asub = GetSubMatrix(A, blockRow, m, wA);
				float *Bsub = GetSubMatrix(B, m, blockCol, wB);

				__shared__ float As[blockSizeMat][blockSizeMat];
				__shared__ float Bs[blockSizeMat][blockSizeMat];

				if ((m * blockSizeMat + col < wA) && (blockRow * blockSizeMat + row < hA))
				{
					As[row][col] = GetElement(Asub, row, col, wA);
				}
				else
				{
					As[row][col] = 0.0;
				}

				if ((m * blockSizeMat + row < wA) && (blockCol * blockSizeMat + col < wB))
				{
					Bs[row][col] = GetElement(Bsub, row, col, wB);
				}
				else
				{
					Bs[row][col] = 0.0;
				}

				__syncthreads();

				for (int e = 0; e < blockSizeMat; ++e)
				{
					Cvalue += As[row][e] * Bs[e][col];
				}

				__syncthreads();
			}

			if ((blockRow * blockSizeMat + row < hA) && (blockCol * blockSizeMat + col < wB))
			{
				SetElement(Csub, row, col, wB, Cvalue);

				//printf("Cvalue: %f \n", Cvalue);
			}
			//else
			//{
			//	printf("Out of range %d %d\n", blockRow * blockSize + row, blockCol * blockSize + col);
			//}
		}


__global__ void trans(const float *A, float *B, int newCol, int newWid)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= newCol*newWid) return;

			int AIdx = index / newWid + (index % newWid * newCol);
			//printf("%d to %d is %d %d to %d %d\n", index, AIdx, index / wid, index % wid, index%wid, index / wid);
			B[index] = A[AIdx];
			//printf("B value: %f \n", B[index]);
		}

		// hA * wA dot wA * wB -> hA * wB
void dot(float *A, float *B, float *C, int hA, int wA, int wB)
		{
			// first transform A and B to matched sizes.

			dim3 dimBlock(blockSizeMat, blockSizeMat);
			dim3 dimGrid((wB + blockSizeMat - 1) / dimBlock.x, (hA + blockSizeMat - 1) / dimBlock.y);

			matDot << <dimGrid, dimBlock >> >(A, B, C, hA, wA, wB);
			cudaDeviceSynchronize();
		}


/*__device__ float GetElementCuda(Matrix2* mat, int row, int col)
		{
			if (row >= *mat->height || col >= *mat->width) return 0;
			return mat->elements[row * (*mat->width) + col];
		}
*/
__device__ float GetElementCuda(Matrix3* mat, int row, int col, int matIdx)
		{
			if (row >= *mat->height || col >= *mat->width || matIdx >= *mat->depth) return 0;
			return mat->elements[matIdx * (*mat->height) * (*mat->width) + row * (*mat->width) + col];
		}

__device__ void SetElementCuda(Matrix2* mat, int row, int col, int value)
		{
			if (row >= (*mat->height) || col >= (*mat->width)) return;
			mat->elements[row * (*mat->width) + col] = value;
		}

__device__ void SetElementCuda(Matrix3* mat, int row, int col, int value, int matIdx)
		{
			if (row >= *mat->height || col >= *mat->width) return;
			(mat->elements + matIdx * (*mat->height) * (*mat->width))[row * (*mat->width) + col] = value;
		}


__global__ void ConvolveTransInputKernel(Matrix3 * input, float* output,
			int patchWidth, int patchHeight,
			int filterWidth, int filterHeight, int stride, int n)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			
			if (index >= n) return;
			int patchSize = filterWidth * filterHeight * (*input->depth);

			int idx = index * patchSize;
			int patchCol = index % patchWidth;
			int patchRow = index / patchWidth;
			int curRow = patchRow * stride;
			int curCol = patchCol * stride;
			//printf("%d %d %d %d %d\n",idx,patchCol,patchRow,curRow,curCol);
			//printf("idx %d patch row %d col %d org row %d col %d\n", idx, patchRow, patchCol, curRow, curCol);
			
			for (int i = 0; i < *input->depth; ++i)
			{
				for (int j = 0; j < filterHeight; ++j)
				{
					for (int k = 0; k < filterWidth; ++k)
					{
						output[idx] = GetElementCuda(input, curRow + j, curCol + k, i);
						//printf("first elements  %f \n",output[idx]);
						idx++;
						
					}
				}
			}
		}

__global__ void ConvolveTransFilterKernel(Matrix3* filters, float* output,
			int patchSize, int filterDepth, int n)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			int idx = index * patchSize;
			int matIdx = index * filterDepth;

			for (int i = 0; i < filterDepth; ++i)
			{
				for (int j = 0; j < *filters->height; ++j)
				{
					for (int k = 0; k < *filters->width; ++k)
					{
						output[idx] = GetElementCuda(filters, j, k, i + matIdx);
						//printf("output %f \n", output[idx]);
						idx++;
					}
				}
			}
		}


void CreateCudaMatrix3(Matrix3 *input, int height, int width, int depth)
		{
			//Matrix3* result = new Matrix3();

			input->depth = (int*)malloc(sizeof(int));
			input->height = (int*)malloc(sizeof(int));
			input->width = (int*)malloc(sizeof(int));
			*input->depth = depth;
			*input->width = width;
			*input->height = height;
			input->elements = (float *)malloc(sizeof(float)*(height * width * depth));

			//printf("size of matrix3 is %d\n size of matrix2 is %d\n", sizeof(Matrix3), sizeof(Matrix2));


			cudaMalloc(&(input->elements), sizeof(float) * height * width * depth);

		}


Matrix3* HostToCudaMatrix3(Matrix3* input)
		{
			Matrix3 *result = (Matrix3*)malloc(sizeof(Matrix3));
			cudaMalloc((void**)&result->height, sizeof(int));  
    		cudaMalloc((void**)&result->width, sizeof(int));  
			cudaMalloc((void**)&result->depth, sizeof(int));
			cudaMalloc(&(result->elements), sizeof(float) * (*input->height) * (*input->width) * (*input->depth));
			cudaMemcpy(result->height, input->height, sizeof(int), cudaMemcpyHostToDevice); 
			cudaMemcpy(result->width, input->width, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(result->depth, input->depth, sizeof(int), cudaMemcpyHostToDevice); 
   
			//CreateCudaMatrix3(result,*input->height, *input->width, *input->depth);

			for (int i = 0; i < *input->depth; ++i)
			{
				cudaMemcpy(result->elements + i * (*input->height) * (*input->width),
					input->mats[i].elements,
					sizeof(float) * (*input->height) * (*input->width), 
					cudaMemcpyHostToDevice);

			}

			return result;
		}

Matrix3* HostToCudaFilter(Matrix3* filters, int filterCount)
		{
			
		    Matrix3 *result = (Matrix3*)malloc(sizeof(Matrix3));
			cudaMalloc((void**)&result->height, sizeof(int));  
    		cudaMalloc((void**)&result->width, sizeof(int));  
			cudaMalloc((void**)&result->depth, sizeof(int));
			cudaMalloc(&(result->elements), sizeof(float) * (*filters->height) * (*filters->width) * (*filters->depth)* filterCount);
			cudaMemcpy(result->height, filters->height, sizeof(int), cudaMemcpyHostToDevice); 
			cudaMemcpy(result->width, filters->width, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(result->depth, filters->depth, sizeof(int), cudaMemcpyHostToDevice); 
			//CreateCudaMatrix3(result,*filters->height, *filters->width, *filters->depth * filterCount);

			int filterMatSize = *filters->height * (*filters->width);

			for (int i = 0; i < filterCount; ++i)
			{
				for (int j = 0; j < *filters->depth; ++j)
				{
					cudaMemcpy(result->elements + i * filterMatSize * (*filters->depth) + j * filterMatSize,
						filters[i].mats[j].elements,
						sizeof(float) * filterMatSize,
						cudaMemcpyHostToDevice);
				}
			}
			//printf("first element%f\n",filters->mats[0].elements[0]);
			return result;
		}


		void FreeCudaMatrix3(Matrix3* input)
		{
			cudaFree(input->elements);
		}





int Convolve(Matrix3* input, Matrix3* filters, int filterCount, int stride)
		{
			


			Matrix3 m3;  
    		cudaMalloc((void**)&m3.height, sizeof(int));  
    		cudaMalloc((void**)&m3.width, sizeof(int));  
			cudaMalloc((void**)&m3.depth, sizeof(int));
			cudaMalloc((void**)&(m3.elements), sizeof(float) * (*input->height) * (*input->width) * (*input->depth));

			Matrix3 *d_input;  
  			cudaMalloc((void**)&d_input, sizeof(Matrix3));  
    		cudaMemcpy(d_input, &m3, sizeof(Matrix3), cudaMemcpyHostToDevice);  
      
    		cudaMemcpy(m3.height, input->height, sizeof(int), cudaMemcpyHostToDevice);  
    		cudaMemcpy(m3.width, input->width, sizeof(int), cudaMemcpyHostToDevice);  
			cudaMemcpy(m3.depth, input->depth, sizeof(int), cudaMemcpyHostToDevice); 
   
			//CreateCudaMatrix3(result,*input->height, *input->width, *input->depth);

			for (int i = 0; i < *input->depth; ++i)
			{
				cudaMemcpy(m3.elements + i * (*input->height) * (*input->width),
					input->mats[i].elements,
					sizeof(float) * (*input->height) * (*input->width), 
					cudaMemcpyHostToDevice);

			}
			int blockSize = 128;
			int height = (*input->height - *filters->height) / stride + 1;
			int width = (*input->width - *filters->width) / stride + 1;
			int patchCount = height * width;
			int patchSize = *filters->width * (*filters->height) * (*input->depth);

//Starting timing
	float gpu_elapsed_time_ms;	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

			float* resultCuda;
			int matSize = patchCount * patchSize;
			cudaMalloc(&resultCuda, sizeof(float) * matSize);

			dim3 dimGrid((patchCount + blockSize - 1) / blockSize);
			//printf("%d, %d", *filters->width, *filters->height);
			
			//printf("%d\n",(patchCount + blockSize - 1) / blockSize);
			ConvolveTransInputKernel <<<dimGrid, blockSize >>>(d_input, resultCuda, 
				width, height,
				*filters->width, *filters->height, stride, patchCount);
			cudaDeviceSynchronize();
			//UTestCheckCudaInputMatT(resultCuda, &input, filterWidth, filterHeight, stride);

			float* resultTransCuda;
			cudaMalloc(&resultTransCuda, sizeof(float) * matSize);

			dim3 fullBlocksPerGrid((matSize + blockSize - 1) / blockSize);
			trans << <fullBlocksPerGrid, blockSize >> >(resultCuda, resultTransCuda,
				patchSize, patchCount);
			//cudaDeviceSynchronize();
			cudaFree(resultCuda);
			
	cudaEventRecord(stop, 0);	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	printf("Input trans Time elapsed on GPU %f ms.\n",gpu_elapsed_time_ms);



			//Matrix3* filterCuda = HostToCudaFilter(filters, filterCount);


			Matrix3 m3_filter;  
    		cudaMalloc((void**)&m3_filter.height, sizeof(int));  
    		cudaMalloc((void**)&m3_filter.width, sizeof(int));  
			cudaMalloc((void**)&m3_filter.depth, sizeof(int));
			cudaMalloc((void**)&(m3_filter.elements), sizeof(float) * (*filters->height) * (*filters->width) * (*filters->depth));

			Matrix3 *d_filter;  
  			cudaMalloc((void**)&d_filter, sizeof(Matrix3));  
    		cudaMemcpy(d_filter, &m3_filter, sizeof(Matrix3), cudaMemcpyHostToDevice);  
      
    		cudaMemcpy(m3_filter.height, filters->height, sizeof(int), cudaMemcpyHostToDevice);  
    		cudaMemcpy(m3_filter.width, filters->width, sizeof(int), cudaMemcpyHostToDevice);  
			cudaMemcpy(m3_filter.depth, filters->depth, sizeof(int), cudaMemcpyHostToDevice); 
   
			//CreateCudaMatrix3(result,*input->height, *input->width, *input->depth);

			int filterMatSize = *filters->height * (*filters->width);

			for (int i = 0; i < filterCount; ++i)
			{
				for (int j = 0; j < *filters->depth; ++j)
				{
					cudaMemcpy(m3_filter.elements + i * filterMatSize * (*filters->depth) + j * filterMatSize,
						filters[i].mats[j].elements,
						sizeof(float) * filterMatSize,
						cudaMemcpyHostToDevice);
				}
			}

			int filterDepth = *filters->depth / filterCount;
			int patchSize_filter = *filters->width * (*filters->height) * filterDepth;
			int patchCount_filter = filterCount;

			float* resultCuda_filter;
			int matSize_filter = patchSize_filter * patchCount_filter;
			cudaMalloc(&resultCuda_filter, sizeof(float) * matSize_filter);

			dim3 dimGrid_filter((patchCount_filter + blockSize - 1) / blockSize);
			//test<<<1,1>>>(d_filter);



	

			ConvolveTransFilterKernel<<<dimGrid_filter, blockSize>>>(d_filter, resultCuda_filter,
				patchSize_filter, filterDepth, filterCount);

			//cudaDeviceSynchronize();


			int height_out = (*input->height - *filters->height) / stride + 1;
			int width_out = (*input->width - *filters->width) / stride + 1;
			int patchCount_out = height_out * width_out;
			int patchSize_out = *filters->width * (*filters->height) * (*filters->depth);

			// Matrix multiplication.
			float *resultCudaOut;
			cudaMalloc(&resultCudaOut, sizeof(float) * filterCount * patchCount_out);






			dot(resultCuda_filter, resultTransCuda, resultCudaOut, filterCount, patchSize_out, patchCount_out);


			
/*
	// Add bias value
			if (bias != nullptr)
			{
				ConvolveAddBias(resultCudaArr, bias, filterCount, patchCount);
			}
*/
			//UTestCheckCudaMat(resultCudaArr, filterCount, patchCount);
			// Convert back to matrix.
			// If we are doing a sequence of computation, then we shouldn't convert back to host memory.

			/*cudaFree(inputCudaArr);
			cudaFree(filterCudaArr);
			cudaFree(inputCuda->elements);
			cudaFree(filterCuda->elements);
*/
			return 0;
		}






