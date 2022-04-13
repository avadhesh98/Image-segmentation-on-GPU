#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define BLOCK_SIZE 16
#define GRID_SIZE 256

__constant__ int centroids_dev;
__constant__ int size_dev;

int CLUSTER_SIZE = 0;
int IMAGE_SIZE = 0;

__constant__ int r_centroid_dev[20];
__constant__ int g_centroid_dev[20];
__constant__ int b_centroid_dev[20];

void setInitialCentroids(int centroids, int* r_centroid, int* g_centroid, int* b_centroid, int red[], int green[], int blue[], int size){
	int randnum;
	srand(time(0));
	for(int i=0;i<centroids;++i)
	{
		randnum = rand()%size;
		r_centroid[i] = red[randnum];
		b_centroid[i] = blue[randnum];
		g_centroid[i] = green[randnum];
	}
}

bool readImage(char* filename, int* red, int* green, int* blue, int size){
	FILE *image;
	image = fopen(filename, "r");
	
	if(image == NULL){
		return false;
	}
	else{
		for(int i=0; i<size; i++){
			red[i] = fgetc(image);
			green[i] = fgetc(image);
			blue[i] = fgetc(image);
		}
		fclose(image);
		return true;
	}
}

bool writeImage(char* filename, int* labels, int* r_centroid, int* g_centroid, int* b_centroid, int size){
	FILE *image;
	image = fopen(filename, "wb");
	
	if(image == NULL){
		return false;
	}
	else{
		for(int i=0; i<size; i++){
			fputc((char) r_centroid[labels[i]],image);
			fputc((char) g_centroid[labels[i]],image);
			fputc((char) b_centroid[labels[i]],image);
		}
		fclose(image);
		return true;
	}
}

__global__ void arrayReset(int* rsum_dev, int* gsum_dev, int* bsum_dev, int* clusterpixelsarray_dev, int* r_centroidtemp_dev, int* g_centroidtemp_dev, int* b_centroidtemp_dev ) {

	// 1 block, 16x16 threads
	int i = threadIdx.x + (threadIdx.y * blockDim.x);

	if(i < centroids_dev) { 
		rsum_dev[i] = 0;
		gsum_dev[i] = 0;
		bsum_dev[i] = 0;
		clusterpixelsarray_dev[i] = 0;
		r_centroidtemp_dev[i] = 0;
		g_centroidtemp_dev[i] = 0;
		b_centroidtemp_dev[i] = 0;
	}
	//__syncthreads();
}

__global__ void labelsReset(int* labels_dev){
	int i = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
	if(i < size_dev) {
		labels_dev[i] = 0;
	}
}

__global__ void labelAssign(int* red_dev, int* green_dev, int* blue_dev, int* labels_dev) {
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
	float min = 500.0, value;
	int index = 0;
	
	if(threadID < size_dev) {
		for(int i = 0; i < centroids_dev; i++) {
			value = sqrtf(powf((red_dev[threadID]-r_centroid_dev[i]),2) + powf((green_dev[threadID]-g_centroid_dev[i]),2) + powf((blue_dev[threadID]-b_centroid_dev[i]),2));

			if(value < min){
				min = value;
				index = i;
			}
		}
		labels_dev[threadID] = index;
	}
	//__syncthreads();
}

__global__ void totalCluster(int* red_dev, int* green_dev, int* blue_dev, int* rsum_dev, int* gsum_dev, int* bsum_dev, int* labels_dev, int* clusterpixelsarray_dev) {

	int i = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

	if(i < size_dev) {
		int currentLabel = labels_dev[i];
		int R = red_dev[i];
		int G = green_dev[i];
		int B = blue_dev[i];
		
		atomicAdd(&rsum_dev[currentLabel], R);
		atomicAdd(&gsum_dev[currentLabel], G);
		atomicAdd(&bsum_dev[currentLabel], B);
		atomicAdd(&clusterpixelsarray_dev[currentLabel], 1);
	}
	//__syncthreads();
}

__global__ void newCentroids(int* r_centroidtemp_dev, int* g_centroidtemp_dev, int* b_centroidtemp_dev, int *rsum_dev, int* gsum_dev, int* bsum_dev, int* clusterpixelsarray_dev, int* flag_dev) {

	int i = threadIdx.x + (threadIdx.y * blockDim.x);
	if(i < centroids_dev) {
		int currentClusterPcount = clusterpixelsarray_dev[i];
		int clustersum_Red = rsum_dev[i];
		int clustersum_Green = gsum_dev[i];
		int clustersum_Blue = bsum_dev[i];
		
		r_centroidtemp_dev[i] = (int)(clustersum_Red/currentClusterPcount);
		g_centroidtemp_dev[i] = (int)(clustersum_Green/currentClusterPcount);
		g_centroidtemp_dev[i] = (int)(clustersum_Blue/currentClusterPcount);
		
		if(r_centroidtemp_dev[i]!=r_centroid_dev[i] || g_centroidtemp_dev[i]!=g_centroid_dev[i] || b_centroidtemp_dev[i]!=b_centroid_dev[i]){
			*flag_dev = 1;}
	}
}

int main(int argc, char* argv[]){
	
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	
	char *input, *output;
	int *red, *green, *blue, *r_centroid, *g_centroid, *b_centroid;
	int *red_dev, *green_dev, *blue_dev, *r_centroidtemp_dev, *g_centroidtemp_dev, *b_centroidtemp_dev;
	int *labels, *labels_dev;
	int width, height, centroids, iterations, size;
	int *clusterpixelsarray, *clusterpixelsarray_dev;
	int *rsum, *gsum, *bsum;
	int flag = 0;
	int *rsum_dev, *gsum_dev, *bsum_dev;
	int *flag_dev;
	
	input = argv[1];
	output = argv[2];
	width = atoi(argv[3]);
	height = atoi(argv[4]);
	centroids = atoi(argv[5]);  
	iterations = atoi(argv[6]);
		
	IMAGE_SIZE = width * height * sizeof(int);
	CLUSTER_SIZE = centroids * sizeof(int);
	size = width * height;
		
	red = (int*)(malloc(IMAGE_SIZE));
	green = (int*)(malloc(IMAGE_SIZE));
	blue = (int*)(malloc(IMAGE_SIZE));
	r_centroid = (int*)(malloc(CLUSTER_SIZE));
	g_centroid = (int*)(malloc(CLUSTER_SIZE));
	b_centroid = (int*)(malloc(CLUSTER_SIZE));
	labels = (int*)(malloc(IMAGE_SIZE)); //stores the cluster number for each pixel
	rsum = (int*)(malloc(CLUSTER_SIZE));
	gsum = (int*)(malloc(CLUSTER_SIZE));
	bsum = (int*)(malloc(CLUSTER_SIZE));
	clusterpixelsarray = (int*)(malloc(CLUSTER_SIZE));

	printf("Loading the image\n");
	if (readImage(input, red, green, blue, size)) {
		printf("Image is loaded");
	} 
	else{
		printf("ERROR: Image could not be loaded");
		return -1;
	}
	
	cudaEvent_t start;
  	cudaEvent_t stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);

	// Setting initial centroids values
	setInitialCentroids(centroids, r_centroid, g_centroid, b_centroid, red, green, blue, size);

	if(IMAGE_SIZE == 0 || CLUSTER_SIZE == 0) {
		return -1;
	}

	// Allocating the memory on device
	cudaMalloc((void**) &red_dev, IMAGE_SIZE);
	cudaMalloc((void**) &green_dev, IMAGE_SIZE);
	cudaMalloc((void**) &blue_dev, IMAGE_SIZE);
	cudaMalloc((void**) &r_centroidtemp_dev, CLUSTER_SIZE);
	cudaMalloc((void**) &g_centroidtemp_dev, CLUSTER_SIZE);
	cudaMalloc((void**) &b_centroidtemp_dev, CLUSTER_SIZE);
	cudaMalloc((void**) &labels_dev, IMAGE_SIZE);
	cudaMalloc((void**) &rsum_dev, CLUSTER_SIZE);
	cudaMalloc((void**) &gsum_dev, CLUSTER_SIZE);
	cudaMalloc((void**) &bsum_dev, CLUSTER_SIZE);
	cudaMalloc((void**) &clusterpixelsarray_dev, CLUSTER_SIZE);
	cudaMalloc((void**) &flag_dev, sizeof(int));

	// copy variables from host to device
	cudaMemcpy(red_dev, red, IMAGE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(green_dev, green, IMAGE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(blue_dev, blue, IMAGE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_centroidtemp_dev, r_centroid, CLUSTER_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(g_centroidtemp_dev, g_centroid, CLUSTER_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(b_centroidtemp_dev, b_centroid, CLUSTER_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(labels_dev, labels, IMAGE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(flag_dev, &flag, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(clusterpixelsarray_dev, clusterpixelsarray, CLUSTER_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(r_centroid_dev, r_centroid, CLUSTER_SIZE);
	cudaMemcpyToSymbol(g_centroid_dev, g_centroid, CLUSTER_SIZE);
	cudaMemcpyToSymbol(b_centroid_dev, b_centroid, CLUSTER_SIZE);
	cudaMemcpyToSymbol(centroids_dev, &centroids, sizeof(int));
	cudaMemcpyToSymbol(size_dev, &size, sizeof(int));

	for(int i = 0; i < centroids; i++) {
		r_centroid[i] = 0;
		g_centroid[i] = 0;
		b_centroid[i] = 0;
	}

	// Defining grid size
	int BLOCK_X, BLOCK_Y;
	BLOCK_X = ceil(width/BLOCK_SIZE);
	BLOCK_Y = ceil(height/BLOCK_SIZE);
	if(BLOCK_X > GRID_SIZE)
		BLOCK_X = GRID_SIZE;
	if(BLOCK_Y > GRID_SIZE)
		BLOCK_Y = GRID_SIZE;

	dim3 dimGRID(BLOCK_X, BLOCK_Y);
	dim3 dimBLOCK(BLOCK_SIZE, BLOCK_SIZE);

	//Starting timer
	cudaEventRecord(start, 0);
	printf("Performing the K means algorithm\n");

	int total_iterations;
	for(int i = 0; i < iterations; i++) {

		total_iterations = i;
		flag=0;
		
		cudaMemcpy(flag_dev, &flag, sizeof(int), cudaMemcpyHostToDevice);
		
		arrayReset<<<1, dimBLOCK>>>(rsum_dev, gsum_dev, bsum_dev, clusterpixelsarray_dev, r_centroidtemp_dev, g_centroidtemp_dev, b_centroidtemp_dev);
		labelsReset<<<dimGRID, dimBLOCK>>>(labels_dev);
		labelAssign<<< dimGRID, dimBLOCK >>> (red_dev, green_dev, blue_dev, labels_dev);
		totalCluster<<<dimGRID, dimBLOCK>>> (red_dev, green_dev, blue_dev, rsum_dev, gsum_dev, bsum_dev, labels_dev, clusterpixelsarray_dev);
		newCentroids<<<1,dimBLOCK >>>(r_centroidtemp_dev, g_centroidtemp_dev, b_centroidtemp_dev, rsum_dev, gsum_dev, bsum_dev, clusterpixelsarray_dev, flag_dev);

		cudaMemcpy(r_centroid, r_centroidtemp_dev, CLUSTER_SIZE, cudaMemcpyDeviceToHost);
		cudaMemcpy(g_centroid, g_centroidtemp_dev, CLUSTER_SIZE, cudaMemcpyDeviceToHost);
		cudaMemcpy(b_centroid, b_centroidtemp_dev, CLUSTER_SIZE, cudaMemcpyDeviceToHost);
		cudaMemcpy(&flag, flag_dev, sizeof(int), cudaMemcpyDeviceToHost);
		
		cudaMemcpyToSymbol(r_centroid_dev, r_centroid, CLUSTER_SIZE);
		cudaMemcpyToSymbol(g_centroid_dev, g_centroid, CLUSTER_SIZE);
		cudaMemcpyToSymbol(b_centroid_dev, b_centroid, CLUSTER_SIZE);

		if(flag==0)
			break;	
	}
	
	cudaEventRecord(stop, 0);
	float elapsed;
	cudaEventSynchronize(stop);
   	cudaEventElapsedTime(&elapsed, start, stop);
		
	cudaMemcpy(labels, labels_dev, IMAGE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(rsum, rsum_dev, CLUSTER_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(gsum, gsum_dev, CLUSTER_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(bsum, bsum_dev, CLUSTER_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(clusterpixelsarray, clusterpixelsarray_dev, CLUSTER_SIZE, cudaMemcpyDeviceToHost);

	printf("Total time taken: %f secs.\n", elapsed/1000.0);
	printf("Converged in %d iterations.\n",total_iterations);
	printf("\n");

	printf("Pixels per centroids:\n");
	for(int k = 0; k < centroids; k++){
		printf("%d centroid: %d pixels\n", k, clusterpixelsarray[k]);
	}

	printf("New centroids:\n");
	for(int i = 0; i < centroids; i++) {
		printf("%d, %d, %d \n", r_centroid[i], g_centroid[i], b_centroid[i]);
	}

	printf("Saving the Image\n");

	if (writeImage(output, labels, r_centroid, g_centroid, b_centroid, size)) {
		printf("Image Saved\n");
	} 
	else{
		printf("ERROR: Unable to save image\n");
		return -1;
	}

	free(red);
	free(green);
	free(blue);
	free(r_centroid);
	free(g_centroid);
	free(b_centroid);
	free(labels);
	free(rsum);
	free(gsum);
	free(bsum);
	free(clusterpixelsarray);

	cudaFree(red_dev);
	cudaFree(green_dev);
	cudaFree(blue_dev);
	cudaFree(r_centroidtemp_dev);
	cudaFree(g_centroidtemp_dev);
	cudaFree(b_centroidtemp_dev);
	cudaFree(labels_dev);
	cudaFree(rsum_dev);
	cudaFree(gsum_dev);
	cudaFree(bsum_dev);
	cudaFree(clusterpixelsarray_dev);
	return 0;
}

