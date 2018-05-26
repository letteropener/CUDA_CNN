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
		};


typedef struct{
			int *B;
}test2;

typedef struct{
			int *A;
			test2 *test;
}test1;



