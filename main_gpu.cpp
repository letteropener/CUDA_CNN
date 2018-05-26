

int main(int argc, char* argv[]){

	const int HEIGHT = 2;
	const int WIDTH = 2;
	const int DEPTH = 2;
	const int FILTER_WIDTH = 1;
	const int FILTER_HEIGHT = 1;

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
	printf("Input Matrix\n");
	
	for	(int i = 0; i < input.depth; ++i){
	input.mats[i].elements = new float[HEIGHT * WIDTH];
		genData(input.mats[i].elements, HEIGHT);
	}
	printMatrix(&input);

	
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
	printf("Filter Matrix\n");
	for	(int i = 0; i < filter.depth; ++i){
		filter.mats[i].elements = new float[filter.height * filter.width];
		genData(filter.mats[i].elements, FILTER_HEIGHT);
	}
	printMatrix(&filter);

	Matrix3* output;	

}
