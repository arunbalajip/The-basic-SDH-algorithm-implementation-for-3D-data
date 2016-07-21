/* ==================================================================
	Programmer: Arunbalaji Prithiviraj (U#80066848) arunbalaji@mail.usf.edu
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc proj2-arunbalaji.cu -o output in the rc machines
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
float *h_x;
float *h_y;
float *h_z;

typedef struct hist_entry{
	unsigned int  d_cnt;   /* need a unsigned int data type as the count might be huge */
} bucket;

bucket * histogram;		/* list of all buckets in the histogram   */
unsigned int	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
float   PDH_res;		/* value of w                             */

/*
	CUDA Kernel. It computes Spatial distance histogram for 3D points. 
	I used concept of shared memory to reduce memory latency in atomic operation. 
	Finally shared memory is loaded to global memory for final result. 
*/
__global__ void SDH (float* x, float*y, float*z ,bucket* hist, const int PDH_acnt, float PDH_res,int num_buckets)
{
	int tx = threadIdx.x;
	int id = blockIdx.x*blockDim.x+tx;
	int j,k;
	int h_pos;
	int half = PDH_acnt/2;
	int odd = PDH_acnt%2;
	extern __shared__ bucket shared_h[];	
	shared_h[tx].d_cnt = 0;
	__syncthreads();
	if(id >= PDH_acnt) return;
	for(k = 1; k <= half; k++)
	{
          	  j=(id+k)%(PDH_acnt);
		h_pos=(int) sqrt((x[id] - x[j])*(x[id]-x[j]) + (y[id] - y[j])*(y[id] - y[j]) + (z[id] - z[j])*(z[id] - z[j]))/PDH_res;
		if(!odd&&k==half&&id>=half) continue;
		atomicAdd(&shared_h[h_pos].d_cnt,1);
	}
	__syncthreads();
	for(k = 0; k<num_buckets&&tx== 0;k++)
		atomicAdd(&hist[k].d_cnt, shared_h[k].d_cnt);
}

/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(bucket* output){
	int i; 
	unsigned int total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15u ", output[i].d_cnt);
		total_cnt += output[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%u \n", total_cnt);
		else printf("| ");
	}
}

int main(int argc, char **argv)
{
	if(argc != 4)
	{
		printf("Missing inputs, Try again!\n");
		exit(0);
	}

	int i;
	bucket* h_histogram;

	/*Device variable declaration*/
	float *d_x, *d_y,*d_z;
	bucket* d_histogram;
	cudaEvent_t start, stop;

	/*command line arguments */
	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	int threads = atoi(argv[3]);
	
	/*Configuration parameters for the device*/
	dim3 dimGrid(ceil(PDH_acnt/threads)+1,1,1);
	dim3 dimBlock(threads,1,1);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;

	/*CPU memory allocation */
	h_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	h_x = (float*)malloc(sizeof(float)*PDH_acnt);
	h_y = (float*)malloc(sizeof(float)*PDH_acnt);
	h_z = (float*)malloc(sizeof(float)*PDH_acnt);

	/*Device memory allocation */
	cudaMalloc(&d_x,sizeof(float)*PDH_acnt);
	cudaMalloc(&d_y,sizeof(float)*PDH_acnt);
	cudaMalloc(&d_z,sizeof(float)*PDH_acnt);
	cudaMalloc(&d_histogram,sizeof(bucket)*num_buckets);
	cudaMemset(d_histogram,0,sizeof(bucket)*num_buckets);
	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		h_x[i] = ((float)(rand()) / RAND_MAX) * BOX_SIZE;
		h_y[i]= ((float)(rand()) / RAND_MAX) * BOX_SIZE;
		h_z[i] = ((float)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	/* Start counting time */
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	/* Copy host array to device*/
	cudaMemcpy(d_x,h_x,sizeof(float)*PDH_acnt,cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,h_y,sizeof(float)*PDH_acnt,cudaMemcpyHostToDevice);
	cudaMemcpy(d_z,h_z,sizeof(float)*PDH_acnt,cudaMemcpyHostToDevice);

	/* Excute the kernel */
	SDH<<<dimGrid,dimBlock,sizeof(bucket)*threads>>>(d_x,d_y,d_z,d_histogram,PDH_acnt,PDH_res,num_buckets);

	/*Copy array back to host */
	cudaMemcpy(h_histogram,d_histogram,sizeof(bucket)*num_buckets,cudaMemcpyDeviceToHost);

	/* check the total running time */
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elaspsedTime;
	cudaEventElapsedTime(&elaspsedTime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/* print out the histogram */
	output_histogram(h_histogram);
	printf("\n******** Total Running Time of Kernel = %0.5f ms *******\n",elaspsedTime);

	/*Release Device Memory*/
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	cudaFree(d_histogram);

	/*Release CPU  Memory*/
	free(h_x);
	free(h_y);
	free(h_z);
	free(h_histogram);
	
	return 0;
}
