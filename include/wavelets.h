//=======================================================================================================
// Copyright 2015 Asgeir Bjorgan, Lise Lyngsnes Randeberg, Norwegian University of Science and Technology
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
// http://opensource.org/licenses/MIT)
//=======================================================================================================

//public api, libwavelet

#ifndef WAVELETS_H_DEFINED
#define WAVELETS_H_DEFINED

#include <vector>
#include <string>

typedef struct{
	//filter properties
	float *filter_h; 
	float *filter_g; 
	int filter_numCoeff;

	//image properties
	int image_bands;
	int image_samples;

	//transformation matrix properties
	int transform_numIterations; //how many batches of details
	float **transform_matrices; //transform matrix for each iteration
	int *transform_numX; //dimensions for each transfmatrix
	int *transform_numY; 
	int transformed_length; //number of bands in the final transform

	//single matrix transform
	float *transform_fullmatrix; //full transformation matrix
	int transform_fullmatrix_numX;
	int transform_fullmatrix_numY;

	//indices to keep
	int keep_numIndices; //number of indices to keep. Corresponds to the number of bands in the output image. 
	int *keep_indices; //the specific indices to keep. Will correspond to the band number in the image as it would have been when all indices had been kept. This array is not allocated when all indices are kept. 

	//CUDA variables
	float *gpu_detail_arr;
	float *gpu_approx_arr_1;
	float *gpu_approx_arr_2;
	float *gpu_filter_h;
	float *gpu_filter_g;
} Wavelet;

//mother wavelet type
enum WaveletType{SYM4};


/** 
 * Wavelet initialization. Wavelet transform will keep all details throughout the specified number of iterations. 
 * \param wavelet Wavelet struct to initialize
 * \param waveletType Type of wavelet (SYM4, ...)
 * \param numIterations Number of iterations in the wavelet transform
 * \param image_bands Number of input image bands
 * \param image_samples Number of input image samples
 **/
void wavelet_initialize(Wavelet *wavelet, WaveletType waveletType, int numIterations, int image_bands, int image_samples);

/**
 * Wavelet initialization. Wavelet transform will only keep the specified subset of details and iterations. 
 * \param wavelet Wavelet struct to initialize
 * \param waveletType Type of wavelet
 * \param numIterations Number of iterations
 * \param indicesToKeep Two-dimensional vector array over which indices to keep. [0][*] corresponds to the details in the first iteration, [1][*] the details in the second iteration, ... . When no details are specified, no details are kept
 * \param image_bands Number of input image bands
 * \param image_samples Number of input image samples
 **/
void wavelet_initialize(Wavelet *wavelet, WaveletType waveletType, int numIterations, std::vector<std::vector<int> > indicesToKeep, int image_bands, int image_samples);

/**
 * Algorithm for calculation of wavelet transform. General advice: Use USE_FILTER_BANK_MATRIX. Sometimes, it can
 * also be advantageous to use USE_FULL_MATRIX, depending on available cache and the size of the input image.
 **/
enum CalcType{USE_FULL_MATRIX, USE_PARTIAL_MATRICES, USE_FILTER_BANK, USE_FILTER_BANK_MATRIX, USE_GPU};

/**
 * Apply wavelet transform to an input line of data.
 * \param wavelet Wavelet properties
 * \param dataLine Line of data
 * \param out_transformed Transformed line of data, allocated within the function
 * \param out_y Number of rows in the output image (i.e. number of bands)
 * \param out_x Number of columns in the output image 
 * \param type Algorithm for calculating the wavelet transform
 **/
void wavelet_run_transform(Wavelet *wavelet, float *dataLine, float **out_transformed, int *out_y, int *out_x, CalcType type = USE_FULL_MATRIX);

/**
 * Get the scaled wavelet function. 
 * \param wavelet Wavelet properties
 * \param iteration Input iteration (zero-indexed)
 * \param detail Input detail (zero indexed)
 * \param wlet Output wavelet function
 **/
void wavelet_get_wavelet_function(Wavelet *wavelet, int iteration, int detail, float *wlet);

/**
 * Get wavelet function corresponding to band in the output image, as it would have been if all indices had been kept.
 * \param wavelet Wavelet properties
 * \param row Band in output image
 * \param wlet Wavelet function
 **/
void wavelet_get_wavelet_function(Wavelet *wavelet, int row, float *wlet);

/**
 * Get iteration and detail corresponding to a specific image output band. 
 * \param wavelet Wavelet properties
 * \param row Band in output image (as it would have been if all indices had been kept)
 * \param iteration Output iteration
 * \param detail Output detail
 **/
void wavelet_get_wavelet_name(Wavelet *wavelet, int row, int *iteration, int *detail);

/**
 * Get band corresponding to specified iteration and detail (as it would have been if all indices had been kept).
 * \param wavelet Wavelet properties
 * \param iteration Input iteration
 * \param detail Input detail
 **/
int wavelet_get_band(Wavelet *wavelet, int iteration, int detail);

/**
 * Free wavelet arrays.
 **/
void wavelet_free(Wavelet *wavelet);


#endif
