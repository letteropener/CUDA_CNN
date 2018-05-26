struct Matrix2 {
			int height;
			int width;
			float *elements;
		};

struct Matrix3 {
			int height;
			int width; 
			int depth;
			Matrix2 *mats;
			float *elements;
		};

Matrix3* Convolve(Matrix3* input, Matrix3* filters, int filterCount, int stride, float* bias=nullptr);
Matrix3* MaxPool(Matrix3* input, int stride);

