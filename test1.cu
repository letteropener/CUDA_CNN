#include <cuda.h>
#include <cuda_runtime.h>
#include "simple.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <string>


__global__ void test(test1* input){
		int idx=threadIdx.x;
		
		printf("index = %d,%d,%d\n",idx,*input->A, *input->test->B); 
}



int main(int argc, char* argv[]){

 
    test1 h_input;  
    h_input.A = (int *)malloc(sizeof(int));  
    h_input.test = (test2 *)malloc(sizeof(test2));  
    h_input.test->B = (int *)malloc(sizeof(int));   
  
    *h_input.A = 5;    
    *h_input.test->B = 3;      
   	//printf("%d,%d\n",h_input.A, h_input.test->B);
    //在显存上定义结构体tmp，使用过渡变量，如果包含多级结构体就需要使用多个过渡变量  
    test1 tmp;  
    cudaMalloc((void**)& tmp.A, sizeof(int));    
    cudaMalloc((void**)& tmp.test, sizeof(test2));  
      
    test2 VN;  
    cudaMalloc((void**)&VN.B, sizeof(int));    
    cudaMemcpy(tmp.test, &VN, sizeof(test2), cudaMemcpyHostToDevice);  
      
  
    test1 *d_input;  
    cudaMalloc((void**)&d_input, sizeof(test1));  
    cudaMemcpy(d_input, &tmp, sizeof(test1), cudaMemcpyHostToDevice);  
      
    //将数据拷贝到显存中，要使用先前定义的过渡变量  
    cudaMemcpy(tmp.A, h_input.A, sizeof(int), cudaMemcpyHostToDevice);    
    cudaMemcpy(VN.B, h_input.test->B, sizeof(int), cudaMemcpyHostToDevice);          
      
    test<<<1, 1>>>(d_input);  
  
    cudaDeviceSynchronize();   
    return 0;

}


