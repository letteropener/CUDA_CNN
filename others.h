struct Matrix2 {
			int *height;
			int *width;
			float *elements;
		};


struct Matrix3 {
			int *height;
			int *width; 
			int *depth;
			Matrix2 *mats;
			float *elements;
		};


int Convolve(Matrix3* input, Matrix3* filters, int filterCount, int stride);
