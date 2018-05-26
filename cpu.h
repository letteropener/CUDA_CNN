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
		};



void genData(float *data, int dim);
float GetElement(Matrix2* mat, int row, int col);
float ConvolveHelp(Matrix3* mat, Matrix3* filter, int row, int col);
Matrix3* Convolve(Matrix3* input, Matrix3* filters, int filterCount, int stride);
Matrix3* MaxPool(Matrix3* input, int stride);
