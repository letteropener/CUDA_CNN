
#include <cstdio>
#include "cpu.cpp"
#include <cmath>
#include <random>
#include <cstdio>
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>


using namespace std;



void printMatrix(Matrix3* mat){
	for (int k = 0; k < mat->depth; ++k)
			{
				for (int i = 0; i < mat->height; ++i)
				{
					for (int j = 0; j < mat->width; ++j)
					{
						printf("%f ", GetElement(&mat->mats[k], i, j));
					}
					printf("\n");
				}

}}

int main(int argc, char* argv[]) {


// initilize timer
  	time_t start,end;
	float time_cost;
	const int HEIGHT = 2048;
	const int WIDTH = 2048;
	const int DEPTH = 3;
	const int FILTER_WIDTH = 3;
	const int FILTER_HEIGHT = 3;

	Matrix3 input;
	input.depth = DEPTH;
	input.width = WIDTH;
	input.height = HEIGHT;
	input.mats = new Matrix2[input.depth];
	for (int i = 0; i < input.depth; ++i)
	{
		input.mats[i].height = input.height;
		input.mats[i].width = input.width;
	}
	//printf("Input Matrix\n");
	
	for	(int i = 0; i < input.depth; ++i){
	input.mats[i].elements = new float[HEIGHT * WIDTH];
		genData(input.mats[i].elements, HEIGHT);
	}
	//printMatrix(&input);

	
	Matrix3 filter;
	filter.depth = DEPTH;
	filter.width = FILTER_WIDTH;
	filter.height = FILTER_HEIGHT;
	filter.mats = new Matrix2[input.depth];
	for (int i = 0; i < filter.depth; ++i)
	{
		filter.mats[i].height = filter.height;
		filter.mats[i].width = filter.width;
	}
	//printf("Filter Matrix\n");
	for	(int i = 0; i < filter.depth; ++i){
		filter.mats[i].elements = new float[filter.height * filter.width];
		genData(filter.mats[i].elements, FILTER_HEIGHT);
	}
	//printMatrix(&filter);
	
	start = clock();
	Matrix3* output;	
	output = Convolve(&input, &filter, 1, 1);
	
	end = clock();
	time_cost = (float)(end-start)/CLOCKS_PER_SEC;	
	printf("the CPU processing time is %f ms\n", 1000*time_cost);	
	
	//printf("Convolve Matrix\n");
	//printMatrix(output);
	//start = clock();
	Matrix3* maxpool;
	maxpool = MaxPool(output, 2);
	//end = clock();
	//time_cost = (float)(end-start)/CLOCKS_PER_SEC;
	//printf("the CPU processing time is %f ms\n", 1000*time_cost);
	
	
	
	//printf("MaxPooling Matrix\n");
	//printMatrix(maxpool);

	

	
	
	return 0;



}
