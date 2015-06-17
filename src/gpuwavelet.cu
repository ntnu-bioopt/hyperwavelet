#include "wavelets.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
using namespace std;





//assume BIP interleaved hyperspectral images
//should have one kernel for each value in the new decimated transformed arrays
//one thread for every value for which we should calculate a convolution
//be sure to input (2*filter_numCoeff + in_approxNumBands + 2*filter_numCoeff)*sizeof(float) as the number of bytes in the shared allocation
//input approximation in in_approx*, filters in filter_h and filter_g, details are output in *out_details and *out_approxs. detail_offset is the number of details to skip (already calculated by previous kernels)
__global__ void wavelet_kernel(float *in_approxArr, int in_approxNumBands, float *filter_h, float *filter_g, int filter_numCoeff, float *out_detailArr, float *out_approxArr, int detail_offset, int total_transform_length){
	int shApproxSize = 2*filter_numCoeff + in_approxNumBands;
	extern __shared__ float sh_arr[]; //dynamic allocation of large shared array containing all three necessary arrays

	//pointers to each part of the array
	float *shApprox = sh_arr;
	float *h = sh_arr + shApproxSize;
	float *g = sh_arr + shApproxSize + filter_numCoeff;
	
	//fill first half of shared reflectance array
	int appInd = threadIdx.x + in_approxNumBands*blockIdx.x; //index in the global approximation array
	int shApproxInd = threadIdx.x + filter_numCoeff; //index in the shared approximation array
	shApprox[shApproxInd] = in_approxArr[appInd];

	//fill second half of shared reflectance array
	appInd = appInd + blockDim.x;
	shApproxInd = shApproxInd + blockDim.x;
	if (shApproxInd < shApproxSize){
		shApprox[shApproxInd] = in_approxArr[appInd];
	}

	//symmetric boundary conditions in the shared reflectance array
	if (threadIdx.x < filter_numCoeff){
		//start of the array
		shApprox[filter_numCoeff - threadIdx.x - 1] = in_approxArr[threadIdx.x + in_approxNumBands*blockIdx.x];

		//end of the array
		shApprox[shApproxSize - threadIdx.x - 1] = in_approxArr[threadIdx.x + in_approxNumBands*blockIdx.x + in_approxNumBands - filter_numCoeff];

		h[threadIdx.x] = filter_h[threadIdx.x];
		g[threadIdx.x] = filter_g[threadIdx.x];
	}
	__syncthreads();

	//reference index from which we take the first value in the convolution	
	appInd = 2*threadIdx.x+1 + filter_numCoeff;

	float conv_h = 0;
	float conv_g = 0;

	//do convolution
	for (int i=0; i < filter_numCoeff; i++){
		float val = shApprox[appInd - i];
		conv_h += val*h[i];
		conv_g += val*g[i];
	}

	//output to approx array so it can be reused
	int outAppInd = threadIdx.x + blockDim.x*blockIdx.x;
	out_approxArr[outAppInd] = conv_h;
	
	//at the same time, output _both_ coefficients to the details array so that there will be less hassle at the last iteration
	int outDetInd = threadIdx.x + total_transform_length*blockIdx.x + detail_offset;
	outAppInd = outDetInd + blockDim.x;
	out_detailArr[outDetInd] = conv_g;
	out_detailArr[outAppInd] = conv_h;
}


void wavelet_initialize_gpu(Wavelet *wavelet){
	cudaMalloc(&(wavelet->gpu_detail_arr), sizeof(float)*wavelet->transformed_length*wavelet->image_samples);
	cudaMalloc(&(wavelet->gpu_approx_arr_1), sizeof(float)*wavelet->transformed_length*wavelet->image_samples);
	cudaMalloc(&(wavelet->gpu_approx_arr_2), sizeof(float)*wavelet->transformed_length*wavelet->image_samples);
	cudaMalloc(&(wavelet->gpu_filter_h), sizeof(float)*wavelet->filter_numCoeff);
	cudaMalloc(&(wavelet->gpu_filter_g), sizeof(float)*wavelet->filter_numCoeff);

	cudaMemcpy(wavelet->gpu_filter_h, wavelet->filter_h, sizeof(float)*wavelet->filter_numCoeff, cudaMemcpyHostToDevice);
	cudaMemcpy(wavelet->gpu_filter_g, wavelet->filter_g, sizeof(float)*wavelet->filter_numCoeff, cudaMemcpyHostToDevice);
}

//convert BIP to BIL or vice versa using cblas
void flip_interleave(float *in, int in_rows, int in_cols, float *out){
	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0;
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, in_rows, in_cols, &alpha, in, in_cols, &beta, in, in_cols, out, in_rows); 
}

void wavelet_run_transformation_gpu(Wavelet *wavelet, float *dataLine, float *out_transformed){
	float *input_approx = wavelet->gpu_approx_arr_1;
	float *output_approx = wavelet->gpu_approx_arr_2;
	int approx_size = wavelet->image_bands;

	float *output_detail = wavelet->gpu_detail_arr;
	int detail_offset = 0;

	cudaMemcpy(input_approx, dataLine, sizeof(float)*wavelet->image_samples*wavelet->image_bands, cudaMemcpyHostToDevice);
	flip_interleave(input_approx, wavelet->image_bands, wavelet->image_samples, output_approx); //BIL to BIP


	for (int i=0; i < wavelet->transform_numIterations; i++){
		//switch arrays
		float *temp = input_approx;
		input_approx = output_approx;
		output_approx = temp;
	
		//calculate grid and sharray size
		size_t shArrSize = (approx_size + 2*wavelet->filter_numCoeff + 2*wavelet->filter_numCoeff)*sizeof(float); //size of shared array allocation: should be space for padded reflectance array and the filters
		int blockSize = wavelet->transform_numY[i]/2;
		dim3 dimGrid = dim3(wavelet->image_samples, 1);
		dim3 dimBlock = dim3(blockSize);

		//run kernel
		wavelet_kernel<<<dimGrid, dimBlock, shArrSize>>>(input_approx, approx_size, wavelet->gpu_filter_h, wavelet->gpu_filter_g, wavelet->filter_numCoeff, output_detail, output_approx, detail_offset, wavelet->transformed_length);
	
		//update offset and next size
		approx_size = wavelet->transform_numY[i]/2;
		detail_offset += approx_size;
	}
	
	//convert interleave again, save to output_approx
	flip_interleave(output_detail, wavelet->image_samples, wavelet->transformed_length, output_approx); //BIP to BIL
	
	//output_approx contains now both details and the remaining approximations, copy to output array
	cudaMemcpy(out_transformed, output_approx, sizeof(float)*wavelet->image_samples*wavelet->transformed_length, cudaMemcpyDeviceToHost);
}
