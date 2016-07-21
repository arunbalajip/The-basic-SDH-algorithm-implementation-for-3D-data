/* ==================================================================
	Programmer: Arunbalaji Prithiviraj (U#80066848) arunbalaji@mail.usf.edu
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc proj1-80066848.cu -o output in the rc machines
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* Thesea are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

/*
	CUDA Kernel. Each Thread takes care of one element of d_atom_list
*/
__global__ void SDH (atom* atomlist,bucket* hist, int PDH_acnt, double PDH_res)
{
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int j,h_pos;
	double dist;
	double x1,x2,y1,y2,z1,z2;
	if(id >= PDH_acnt) return;
	for(j = id+1; j < PDH_acnt; j++)
	{
		x1 = atomlist[id].x_pos;
		x2 = atomlist[j].x_pos;
		y1 = atomlist[id].y_pos;
		y2 = atomlist[j].y_pos;
		z1 = atomlist[id].z_pos;
		z2 = atomlist[j].z_pos;
		dist = sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
		h_pos = (int) (dist / PDH_res);
		atomicAdd(&hist[h_pos].d_cnt,1);
	}
}
		


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(bucket* output){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", output[i].d_cnt);
		total_cnt += output[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


int main(int argc, char **argv)
{
	int i;
	bucket* h_histogram;

	/*Device variable declaration*/
	atom* d_atom_list;
	bucket* d_histogram;

	/*command line arguments */
	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);

	/*Configuration parameters for the device*/
	dim3 dimGrid(ceil(PDH_acnt/512)+1,1,1);
	dim3 dimBlock(512,1,1);

//printf("args are %d and %f\n", PDH_acnt, PDH_res);
	/*CPU memory allocation */
	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
        histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	h_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	/*Device memory allocation */
	cudaMalloc(&d_atom_list,sizeof(atom)*PDH_acnt);
	cudaMalloc(&d_histogram,sizeof(bucket)*num_buckets);

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* Copy host array to device*/
	cudaMemcpy(d_atom_list,atom_list,sizeof(atom)*PDH_acnt,cudaMemcpyHostToDevice);
    	cudaMemcpy(d_histogram,histogram,sizeof(bucket)*num_buckets,cudaMemcpyHostToDevice);

	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time();
	
	/* print out the histogram */
	output_histogram(histogram);
	
	/* Excute the kernel */
	SDH<<<dimGrid,dimBlock>>>(d_atom_list,d_histogram,PDH_acnt,PDH_res);

	/*Copy array back to host */
	cudaMemcpy(h_histogram,d_histogram,sizeof(bucket)*num_buckets,cudaMemcpyDeviceToHost);

	/* print out the histogram */
	printf("\nGPU version:"); 
	output_histogram(h_histogram);

	/*Release Device Memory*/
	cudaFree(d_atom_list);
	cudaFree(d_histogram);

	/*Release CPU  Memory*/
	free(atom_list);
	free(histogram);
	free(h_histogram);
	
	return 0;
}
