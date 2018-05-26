#include <cuda.h>
#include <cuda_runtime.h>
#include "simple.cu"
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


#define gpuErrchk(ans){gpuAssert((ans),__FILE__,__LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false){
	if(code != cudaSuccess){
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
		if(abort) exit(code);	
	}else{
		//printf("cuda returned code == cudaSuccess\n");	
	}
}

Matrix3* matrix3movetoCuda(Matrix3 mat_cpu){
	int mat_height = *mat_cpu.height;
	int mat_width = *mat_cpu.width;
	Matrix3 m3;  
    cudaMalloc((void**)&m3.height, sizeof(int));  
    cudaMalloc((void**)&m3.width, sizeof(int));  
	cudaMalloc((void**)&m3.depth, sizeof(int));
    cudaMalloc((void**)&m3.mats, sizeof(Matrix2)*3);  
      
    Matrix2 m2_1;  
    cudaMalloc((void**)&m2_1.height, sizeof(int));  
    cudaMalloc((void**)&m2_1.width, sizeof(int));
	cudaMalloc((void**)&m2_1.elements, sizeof(float)*mat_height*mat_width);
	Matrix2 m2_2;  
    cudaMalloc((void**)&m2_2.height, sizeof(int));  
    cudaMalloc((void**)&m2_2.width, sizeof(int));
	cudaMalloc((void**)&m2_2.elements, sizeof(float)*mat_height*mat_width);
	Matrix2 m2_3;  
    cudaMalloc((void**)&m2_3.height, sizeof(int));  
    cudaMalloc((void**)&m2_3.width, sizeof(int));
	cudaMalloc((void**)&m2_3.elements, sizeof(float)*mat_height*mat_width);    

	cudaMemcpy(&m3.mats[0], &m2_1, sizeof(Matrix2), cudaMemcpyHostToDevice);  
	cudaMemcpy(&m3.mats[1], &m2_2, sizeof(Matrix2), cudaMemcpyHostToDevice);  
	cudaMemcpy(&m3.mats[2], &m2_3, sizeof(Matrix2), cudaMemcpyHostToDevice);  
      
  
    Matrix3 *d_input;  
    cudaMalloc((void**)&d_input, sizeof(Matrix3));  
    cudaMemcpy(d_input, &m3, sizeof(Matrix3), cudaMemcpyHostToDevice);  
      
    cudaMemcpy(m3.height, mat_cpu.height, sizeof(int), cudaMemcpyHostToDevice);  
    cudaMemcpy(m3.width, mat_cpu.width, sizeof(int), cudaMemcpyHostToDevice);  
	cudaMemcpy(m3.depth, mat_cpu.depth, sizeof(int), cudaMemcpyHostToDevice); 
    
	cudaMemcpy(m2_1.height, mat_cpu.mats[0].height, sizeof(int), cudaMemcpyHostToDevice);     
    cudaMemcpy(m2_1.width, mat_cpu.mats[0].width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m2_1.elements, mat_cpu.mats[0].elements, sizeof(float)*mat_height*mat_width, cudaMemcpyHostToDevice); 
	
	cudaMemcpy(m2_2.height, mat_cpu.mats[1].height, sizeof(int), cudaMemcpyHostToDevice);     
    cudaMemcpy(m2_2.width, mat_cpu.mats[1].width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m2_2.elements, mat_cpu.mats[1].elements, sizeof(float)*mat_height*mat_width, cudaMemcpyHostToDevice); 

	cudaMemcpy(m2_3.height, mat_cpu.mats[2].height, sizeof(int), cudaMemcpyHostToDevice);     
    cudaMemcpy(m2_3.width, mat_cpu.mats[2].width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m2_3.elements, mat_cpu.mats[2].elements, sizeof(float)*mat_height*mat_width, cudaMemcpyHostToDevice);   
	return d_input;

}

Matrix3* matrix3movetoCuda_ouput(Matrix3 mat_cpu){
	int mat_height = *mat_cpu.height;
	int mat_width = *mat_cpu.width;
	Matrix3 m3;  
    cudaMalloc((void**)&m3.height, sizeof(int));  
    cudaMalloc((void**)&m3.width, sizeof(int));  
	cudaMalloc((void**)&m3.depth, sizeof(int));
    cudaMalloc((void**)&m3.mats, sizeof(Matrix2));  
      
    Matrix2 m2_1;  
    cudaMalloc((void**)&m2_1.height, sizeof(int));  
    cudaMalloc((void**)&m2_1.width, sizeof(int));
	cudaMalloc((void**)&m2_1.elements, sizeof(float)*mat_height*mat_width);
	cudaMemcpy(&m3.mats[0], &m2_1, sizeof(Matrix2), cudaMemcpyHostToDevice);  

	Matrix3 *d_input;  
    cudaMalloc((void**)&d_input, sizeof(Matrix3));  
    cudaMemcpy(d_input, &m3, sizeof(Matrix3), cudaMemcpyHostToDevice);  

    cudaMemcpy(m3.height, mat_cpu.height, sizeof(int), cudaMemcpyHostToDevice);  
    cudaMemcpy(m3.width, mat_cpu.width, sizeof(int), cudaMemcpyHostToDevice);  
	cudaMemcpy(m3.depth, mat_cpu.depth, sizeof(int), cudaMemcpyHostToDevice); 

	cudaMemcpy(m2_1.height, mat_cpu.mats[0].height, sizeof(int), cudaMemcpyHostToDevice);     
    cudaMemcpy(m2_1.width, mat_cpu.mats[0].width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m2_1.elements, mat_cpu.mats[0].elements, sizeof(float)*mat_height*mat_width, cudaMemcpyHostToDevice); 
  
    
      
 
	return d_input;

}



int main(int argc, char* argv[]){

	const int HEIGHT = 2048;
	const int WIDTH = 2048;
	const int DEPTH = 3;
	const int FILTER_WIDTH = 3;
	const int FILTER_HEIGHT = 3;
	const int MAXPOOLING_STRIDE = 2;
//initial input Matrix in host
	Matrix3 input;
	input.depth = (int*)malloc(sizeof(int));
	input.height = (int*)malloc(sizeof(int));
	input.width = (int*)malloc(sizeof(int));
	*input.depth = DEPTH;
	*input.width = WIDTH;
	*input.height = HEIGHT;
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
//initial output Matrix in host	
	Matrix3 output;
	output.depth = (int*)malloc(sizeof(int));
	output.height = (int*)malloc(sizeof(int));
	output.width = (int*)malloc(sizeof(int));
	*output.depth = 1;
	*output.height = (HEIGHT - FILTER_HEIGHT) / 1 + 1;
	*output.width = (WIDTH - FILTER_WIDTH) / 1 + 1;
	output.mats = (Matrix2*)malloc(sizeof(Matrix2));
	
	for (int i = 0; i < 1; ++i)
	{
		output.mats[i].width = (int*)malloc(sizeof(int));
		output.mats[i].height = (int*)malloc(sizeof(int));
		*output.mats[i].height = *output.height;
		*output.mats[i].width = *output.width;
	} 
	//printf("Output Matrix completed\n");
	for	(int i = 0; i < DEPTH; ++i){
		output.mats[i].elements = (float*)malloc(sizeof(float)*((HEIGHT - FILTER_HEIGHT) / 1 + 1)*((WIDTH - FILTER_WIDTH) / 1 + 1));
	}


// initial output MaxPooling Matrix in host
	Matrix3 Output_MaxPooling;
	Output_MaxPooling.depth = (int*)malloc(sizeof(int));
	Output_MaxPooling.height = (int*)malloc(sizeof(int));
	Output_MaxPooling.width = (int*)malloc(sizeof(int));
	*Output_MaxPooling.depth = 1;
	*Output_MaxPooling.height = ((HEIGHT - FILTER_HEIGHT) / 1 + 1)% MAXPOOLING_STRIDE == 0 ? 
	((HEIGHT - FILTER_HEIGHT) / 1 + 1)/ MAXPOOLING_STRIDE :	
	 (((HEIGHT - FILTER_HEIGHT) / 1 + 1)/ MAXPOOLING_STRIDE)+1;
	*Output_MaxPooling.width = ((WIDTH - FILTER_WIDTH) / 1 + 1)% MAXPOOLING_STRIDE == 0 ?
 	((WIDTH - FILTER_WIDTH) / 1 + 1)/ MAXPOOLING_STRIDE:
	(((WIDTH - FILTER_WIDTH) / 1 + 1)/ MAXPOOLING_STRIDE)+1;
	Output_MaxPooling.mats = (Matrix2*)malloc(sizeof(Matrix2));
	
	for (int i = 0; i < 1; ++i)
	{
		Output_MaxPooling.mats[i].width = (int*)malloc(sizeof(int));
		Output_MaxPooling.mats[i].height = (int*)malloc(sizeof(int));
		*Output_MaxPooling.mats[i].height = *Output_MaxPooling.height;
		*Output_MaxPooling.mats[i].width = *Output_MaxPooling.width;
	} 
	//printf("Output MaxPooling Matrix completed\n");
	for	(int i = 0; i < DEPTH; ++i){
		Output_MaxPooling.mats[i].elements = (float*)malloc(sizeof(float)*(*Output_MaxPooling.height)*(*Output_MaxPooling.width));
	}
	
//Starting timing
	float gpu_elapsed_time_ms;	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
//initiate cuda Matrix	
	Matrix3 *filter_GPU,*input_GPU,*output_GPU, *Output_MaxPooling_GPU;
	
	filter_GPU = matrix3movetoCuda(filter);
	input_GPU = matrix3movetoCuda(input);

//create cuda output Matrix
	output_GPU = matrix3movetoCuda_ouput(output);

//create cuda maxpooling output Matrix
	Output_MaxPooling_GPU = matrix3movetoCuda_ouput(Output_MaxPooling);

//initiate cuda thread parameter
	int blockSize = 16;
	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(*output.width / blockSize + 1, *output.height / blockSize + 1);

	//printf("mat max %d, grid max %d\n", dimGrid.x, dimGrid.y);

	ConvolveKernel <<<dimGrid, dimBlock >>>(input_GPU,output_GPU,filter_GPU, 1,1, blockSize);
	//ConvolveKernel_SharedMemory<<<dimGrid, dimBlock >>>(input_GPU,output_GPU,filter_GPU, 1,1, blockSize);
	//printf("convlve finished !\n");
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());	
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	//printf("Time elapsed on GPU %f ms.\n",gpu_elapsed_time_ms);
	printf("No shared mem Time elapsed on GPU %f ms.\n",gpu_elapsed_time_ms);

	
	cudaEventRecord(start, 0);
//initiate cuda Matrix	
	Matrix3 *filter_GPU_shared,*input_GPU_shared,*output_GPU_shared,*Output_MaxPooling_GPU_shared;
	
	filter_GPU_shared = matrix3movetoCuda(filter);
	input_GPU_shared = matrix3movetoCuda(input);

//create cuda output Matrix
	output_GPU_shared = matrix3movetoCuda_ouput(output);

//create cuda maxpooling output Matrix
	Output_MaxPooling_GPU_shared = matrix3movetoCuda_ouput(Output_MaxPooling);

//initiate cuda thread parameter
	dim3 dimBlock_shared(blockSize, blockSize);
	dim3 dimGrid_shared(*output.width / blockSize + 1, *output.height / blockSize + 1);
	ConvolveKernel_SharedMemory <<<dimGrid_shared, dimBlock_shared >>>(input_GPU_shared,output_GPU_shared,filter_GPU_shared, 1,1, blockSize);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	printf("with shared mem Time elapsed on GPU %f ms.\n",gpu_elapsed_time_ms);

	//cudaEventRecord(start, 0);
	int blockSize_Max = 16;
	dim3 dimBlock_Max(blockSize_Max, blockSize_Max);
	dim3 dimGrid_Max(*Output_MaxPooling.width / blockSize_Max + 1, *Output_MaxPooling.height / blockSize_Max + 1);
	//printf("%d\n",*Output_MaxPooling.width);
	MaxPoolKernel<<<dimGrid_Max, dimBlock_Max>>>(output_GPU, Output_MaxPooling_GPU, 2, blockSize_Max);
	//printf("After MaxPooling ! \n");
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());	
	cudaDeviceSynchronize();
 	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	//printf("Time elapsed on GPU %f ms.\n", gpu_elapsed_time_ms);	


	Matrix3 m3;  
    cudaMemcpy(&m3, Output_MaxPooling_GPU, sizeof(Matrix3), cudaMemcpyDeviceToHost);  
    Matrix2 m2;  
    cudaMemcpy(&m2, m3.mats, sizeof(Matrix2), cudaMemcpyDeviceToHost);  
  
    cudaMemcpy(Output_MaxPooling.mats->elements, m2.elements, sizeof(float)*(*Output_MaxPooling.height)*(*Output_MaxPooling.width), cudaMemcpyDeviceToHost);    
	

	
        //printMatrix(&Output_MaxPooling);
	cudaFree(input_GPU);
	cudaFree(output_GPU);
	cudaFree(filter_GPU);
	
	return 0;
}

