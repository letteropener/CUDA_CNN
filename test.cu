#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>



#define gpuErrchk(ans){gpuAssert((ans),__FILE__,__LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false){
	if(code != cudaSuccess){
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
		if(abort) exit(code);	
	}else{
		printf("cuda returned code == cudaSuccess\n");	
	}
}

void MatrixMulCPU(float* C, float* A,float* B, int n){
	float sum = 0;
	for(int i = 0;i < n; i++){
		for(int j = 0;j < n; j++){
			sum = 0;
			for(int k = 0; k < n; k++){
				sum += A[i*n + k]*B[k*n + j];			
			}
		C[i*n+j] = sum;			
		}	
	}
}

void matgen(float* a, int n){
	int i,j;
	for(i = 0;i<n; i++){
		for(j =0; j<n ;j++){
			a[i*n + j] = (float)rand()/RAND_MAX;		
		}	
	}
}

__global__ void matrixMulCUDA(float *a, float *b, float *c, int n){

	int Bx = blockIdx.x;
	int Tx = threadIdx.x;
	int column = Bx * blockDim.x + Tx;
	
	int By = blockIdx.y;
	int Ty = threadIdx.y;
	int row = By * blockDim.y + Ty;



	if(row<n && column <n){
		float sum=0;
		for(int i = 0; i < n; i++){
			sum += a[row * n + i] * b[i*n + column];
		} 
		c[row*n+column] = sum;
	}
}

int main(){
	time_t start,end;
	float time_cost;
	float *a,*b,*c;
	float *d_a,*d_b,*d_c;
	
	/*matrix width*/
	int n = 1000;
	/*block width*/
	int blockwidth =10;

	a = (float*)malloc(n*n*sizeof(float));
	b = (float*)malloc(n*n*sizeof(float));
	c = (float*)malloc(n*n*sizeof(float));
	srand(0);
	
	matgen(a,n);
	matgen(b,n);

	start = clock();

	cudaMalloc((void**)&d_a,sizeof(float)*n*n);
	cudaMalloc((void**)&d_b,sizeof(float)*n*n);
	cudaMalloc((void**)&d_c,sizeof(float)*n*n);
	
	cudaMemcpy(d_a,a,sizeof(float)*n*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,sizeof(float)*n*n,cudaMemcpyHostToDevice);
	
	dim3 blockdim(blockwidth,blockwidth,1);
	dim3 griddim(n/blockwidth+1,n/blockwidth+1,1);
	
	matrixMulCUDA<<<griddim, blockdim>>>(d_a, d_b, d_c, n);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaDeviceSynchronize();
	
	cudaMemcpy(c,d_c,sizeof(float)*n*n,cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	end = clock();
	time_cost = (double)(end-start)/CLOCKS_PER_SEC;
	printf("the GPU processing time is %f s\n",time_cost);
	
	start = clock();
	MatrixMulCPU(c, a, b, n);
	end = clock();
	time_cost = (double)(end-start)/CLOCKS_PER_SEC;
	printf("the CPU processing time is %f s\n",time_cost);
	return 0;
}
