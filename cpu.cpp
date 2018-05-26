#include "cpu.h"
#include <cmath>
#include <random>
#include <cstdio>
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;



void genData(float *data, int dim)
{
	for (int i = 0; i < dim*dim; ++i)
	{
		data[i] = (((float)rand() / (float)RAND_MAX) - 0.5f) * 10;
	}
}


float GetElement(Matrix2* mat, int row, int col)
		{
			if (row >= mat->height || col >= mat->width) return 0;
			return mat->elements[row * mat->width + col];
		}

float ConvolveHelp(Matrix3* mat, Matrix3* filter, int row, int col)
		{
			if (row > mat->height || col > mat->width) return 0;
			float value = 0.0;
			for (int k = 0; k < mat->depth; ++k)
			{
				for (int i = 0; i < filter->height; ++i)
				{
					for (int j = 0; j < filter->width; ++j)
					{
						value += GetElement(&mat->mats[k], row + i, col + j) * GetElement(&filter->mats[k], i, j);
					}
				}
			}
			//printf("value %f \n\n", value);
			return value;
		}



Matrix3* Convolve(Matrix3* input, Matrix3* filters, int filterCount, int stride)
		{
			Matrix3* mat3 = new Matrix3();
			mat3->depth = filterCount;
			mat3->height = (input->height - filters[0].height) / stride + 1;
			mat3->width = (input->width - filters[0].width) / stride + 1;
			mat3->mats = new Matrix2[filterCount];
			for (int i = 0; i < filterCount; ++i)
				{
					mat3->mats[i].height = mat3->height;
					mat3->mats[i].width = mat3->width;
				}
		
			/*printf("\n result:%d\t",mat3->mats[0].height);*/

			int rowIdx = 0, colIdx = 0;
			for (int i = 0; i < filterCount; ++i)
			{
				colIdx = 0;
				mat3->mats[i].elements = new float[mat3->height * mat3->width];
				// convolve 
				for (int j = 0; j + filters[i].width <= input->width; j += stride)
				{
					rowIdx = 0;
					for (int k = 0; k + filters[i].height <= input->height; k += stride)
					{
						int idx = colIdx + rowIdx * mat3->width;
						mat3->mats[i].elements[idx] = ConvolveHelp(input, &filters[i], k, j);
						rowIdx++;
						/*printf("\n%f\t",mat3->mats[i].elements[idx] );*/
					}
					colIdx++;
				}
			}
			
			return mat3;
		}

Matrix3* MaxPool(Matrix3* input, int stride)
		{
			Matrix3* result = new Matrix3();
			result->depth = input->depth;
			result->width = (input->width + stride - 1) / stride;
			result->height = (input->height + stride - 1) / stride;
			result->mats = new Matrix2[input->depth];

			for (int i = 0; i < input->depth; ++i)
			{
				result->mats[i].height = result->height;
				result->mats[i].width = result->width;
				result->mats[i].elements = new float[result->width * result->height];

				for (int j = 0; j < result->width; ++j)
				{
					for (int k = 0; k < result->height; ++k)
					{
						float value = 0;
						for (int m = 0; m < stride; ++m)
						{
							for (int n = 0; n < stride; ++n)
							{
								float cur_v = GetElement(&input->mats[i], k*stride + n, j*stride + m);
								//printf("get %d %d of value %f\n", k*stride + n, j*stride + m, cur_v);
								if (value < cur_v)
								{
									value = cur_v;
								}
							}
						}
						//printf("assign %d %d of value %f\n", k, j, value);
						result->mats[i].elements[k*result->width + j] = value;
					}
				}
			}

			return result;
		}
