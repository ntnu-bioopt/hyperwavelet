//=======================================================================================================
// Copyright 2015 Asgeir Bjorgan, Lise Lyngsnes Randeberg, Norwegian University of Science and Technology
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
// http://opensource.org/licenses/MIT)
//=======================================================================================================

#include <iostream>
#include <string.h>
#include "wavelets.h"
#include "wavelets_private.h"
#include <cblas.h>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

//helper function for decimation
bool saveToMatrix(int i){
	return ((i % 2) != 0);
}

void wavelet_transform_matrix(float *filter_h, int filter_numCoeffs, int image_bands, float **out_transf, int *out_numY, int *out_numX){ 
	int transform_numX, transform_numY;
	transform_numX = image_bands;
	transform_numY = (image_bands + filter_numCoeffs - 1)/2; //#bands values (with 7 boundconds), then filter_numCoeffs-1 places beyond the end of the array in order to include the last value
	float *transf = new float[transform_numX * transform_numY]();


	//symmetric boundary conditions for the transformation (transformation of signal value nr 0 to nr. filter_numCoeff - 1, and of the end of the signal array)
	int *indices = new int[filter_numCoeffs]; //index at which the corresponding coefficient is placed at any time
	for (int i=0; i < filter_numCoeffs; i++){
		indices[i] = i-1; //initialized in the opposite direction
	}
	indices[0] = 0; //h0+h1 in the beginning
	int decrInd = 1;
	int row = 0;
	int i=0;

	for (i=0; i < filter_numCoeffs; i++){
		bool shouldSaveToMatrix = saveToMatrix(i); //decimation

		//set coefficients
		if (shouldSaveToMatrix){
			for (int j=0; j < filter_numCoeffs; j++){
				transf[row*transform_numX + indices[j]] += filter_h[j];
			}
			row++;
		}
		
		//update indices for reflection

		//increment indices
		for (int k=0; k < decrInd; k++){
			indices[k]++;
		}

		//decrement indices
		for (int k=decrInd; k < filter_numCoeffs; k++){
			indices[k]--;
			if (indices[k] < 0){
				indices[k] = 0;
			}
		}
		decrInd++;
	}
	
	//transformation matrix values well within the image band ranges
	int offset = 1;
	int start = i;
	for (i=start; i < image_bands; i++){
		bool shouldSaveToMatrix = saveToMatrix(i); //decimation
		
		if (shouldSaveToMatrix){
			for (int j=0; j < filter_numCoeffs; j++){
				transf[row*transform_numX + j + offset] = filter_h[filter_numCoeffs - 1 - j];
			}
			row++;
		}
		offset++;
	}

	//boundary conditions at the end of the signal array
	for (int i=0; i < filter_numCoeffs; i++){
		indices[i] = filter_numCoeffs-i; //initialized in the opposite direction
	}
	indices[0] = filter_numCoeffs-1; //h0+h1 in the end
	decrInd = 1;
	offset = offset-1; //adjust for the last increment
	start = i;

	for (i=start; i < filter_numCoeffs + image_bands; i++){
		bool shouldSaveToMatrix = saveToMatrix(i); //decimation

		//set coefficients
		if (shouldSaveToMatrix && (row < transform_numY)){
			for (int j=0; j < filter_numCoeffs; j++){
				transf[row*transform_numX + indices[j] + offset] += filter_h[j];
			}
			row++;
		}
		
		//update indices for reflection

		//increment indices
		for (int k=0; k < decrInd; k++){
			indices[k]--;
		}

		//decrement indices
		for (int k=decrInd; k < filter_numCoeffs; k++){
			indices[k]++;
			if (indices[k] >= filter_numCoeffs){
				indices[k] = filter_numCoeffs-1;
			}
		}
		decrInd++;
	}

	
	delete [] indices;

	//output
	*out_numX = transform_numX;
	*out_numY = transform_numY;
	*out_transf = transf;
}



void wavelet_generate_transformation(Wavelet *wavelet, int signal_length, float **out_transf, int *out_numY, int *out_numX){
	//approximation
	int approx_transform_numY, approx_transform_numX;
	float *approx_transf;
	wavelet_transform_matrix(wavelet->filter_h, wavelet->filter_numCoeff, signal_length, &approx_transf, &approx_transform_numY, &approx_transform_numX);
	
	//detail
	int detail_transform_numY, detail_transform_numX;
	float *detail_transf;
	wavelet_transform_matrix(wavelet->filter_g, wavelet->filter_numCoeff, signal_length, &detail_transf, &detail_transform_numY, &detail_transform_numX);

	//combine to one transformation matrix
	int transform_numX = approx_transform_numX;
	int transform_numY = approx_transform_numY + detail_transform_numY;
	float *transf = new float[transform_numX * transform_numY];
	memcpy(transf, detail_transf, sizeof(float)*detail_transform_numY*transform_numX);
	memcpy(transf + detail_transform_numY*transform_numX, approx_transf, sizeof(float)*approx_transform_numY*transform_numX);
	
	delete [] detail_transf;
	delete [] approx_transf;

	//output
	*out_transf = transf;
	*out_numX = transform_numX;
	*out_numY = transform_numY;
}	

void wavelet_initialize(Wavelet *wavelet, WaveletType waveletType, int numIterations, vector<vector<int> > keepIndices, int image_bands, int image_samples){
	wavelet_initialize(wavelet, waveletType, numIterations, image_bands, image_samples);

	//create index table
	vector<int> indices;
	int skipIndices = 0;
	for (unsigned int i=0; i < keepIndices.size(); i++){
		for (unsigned int j=0; j < keepIndices[i].size(); j++){
			indices.push_back(skipIndices + keepIndices[i][j]);
		}
		skipIndices += wavelet->transform_numY[i]/2;
		cerr << wavelet->transform_numY[i]/2 << " ";
	}
	cerr << endl;
	wavelet->keep_numIndices = indices.size();
	wavelet->keep_indices = new int[wavelet->keep_numIndices];
	copy(indices.begin(), indices.end(), wavelet->keep_indices);

	cerr << "Kept indices in wavelet transform: ";
	for (int i=0; i < wavelet->keep_numIndices; i++){
		cerr << wavelet->keep_indices[i] << " ";
	}
	cerr << endl;
}

void wavelet_initialize(Wavelet *wavelet, WaveletType waveletType, int numIterations, int image_bands, int image_samples){
	wavelet->transform_numIterations = numIterations;
	wavelet->image_samples = image_samples;
	wavelet->image_bands = image_bands;
	switch (waveletType){
		case SYM4:
			wavelet->filter_numCoeff = SYM4_NUMCOEFF;
			wavelet->filter_h = new float[wavelet->filter_numCoeff];
			wavelet->filter_g = new float[wavelet->filter_numCoeff];
			for (int i=0; i < wavelet->filter_numCoeff; i++){
				wavelet->filter_h[i] = SYM4_H[i];
				wavelet->filter_g[i] = SYM4_G[i];
			}
		break;
		default:
			cerr << "Unknown wavelet type" << endl;
			exit(1);
		break;	
	}

	//create transformation matrices, one for each iteration
	wavelet->transform_matrices = new float*[numIterations];
	wavelet->transform_numX = new int[numIterations];
	wavelet->transform_numY = new int[numIterations];
	int signalLength = wavelet->image_bands;
	wavelet->transformed_length = 0;
	for (int i=0; i < wavelet->transform_numIterations; i++){
		float *temp_transf = NULL;
		wavelet_generate_transformation(wavelet, signalLength, &temp_transf, &(wavelet->transform_numY[i]), &(wavelet->transform_numX[i]));
		wavelet->transform_matrices[i] = temp_transf;

		signalLength = wavelet->transform_numY[i]/2; //length of approximation part of the signal array
		wavelet->transformed_length += signalLength;
		fprintf(stderr, "%d\n", signalLength);
	}
	wavelet->transformed_length += signalLength; //last approximation part
	
	wavelet->keep_numIndices = wavelet->transformed_length;
	wavelet->keep_indices = NULL; //not used when we keep all of the indices

	//create total transformation matrix for each iteration as if it was run on the [details | approximation] array and not only the approximation array, and combine them through matrix multiplication
	float **transform_matrices = new float*[wavelet->transform_numIterations];
	int *num_x = new int[wavelet->transform_numIterations];
	int *num_y = new int[wavelet->transform_numIterations];
	for (int i=0; i < wavelet->transform_numIterations; i++){
		wavelet_generate_fulltransformationmat(wavelet, i, &transform_matrices[i], &(num_y[i]), &(num_x[i]));
	}

	//combine transformation matrices by multiplication
	int prevY = num_y[0];
	int prevX = num_x[0];
	float *prevCombinedTransf = new float[prevX*prevY];
	memcpy(prevCombinedTransf, transform_matrices[0], sizeof(float)*prevY*prevX);
	
	for (int i=1; i < wavelet->transform_numIterations; i++){
		int newX = prevX; // (y1 x x1) x (y0 x x0) => y1 x x0
		int newY = num_y[i];

		float *newTransf = new float[newX*newY];

		//multiply
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_y[i], prevX, num_x[i], 1.0f, transform_matrices[i], num_x[i], prevCombinedTransf, prevX, 0.0f, newTransf, newX);

		delete [] prevCombinedTransf;
		prevCombinedTransf = newTransf;
		prevX = newX;
		prevY = newY;
	}

	//combined transformation matrix is now ready
	wavelet->transform_fullmatrix = prevCombinedTransf;
	wavelet->transform_fullmatrix_numX = prevX;
	wavelet->transform_fullmatrix_numY = prevY;

	//cleanup
	delete [] num_y;
	delete [] num_x;
	for (int i=0; i < numIterations; i++){
		delete [] transform_matrices[i];
	}
	delete [] transform_matrices;

	//gpu arrays
	wavelet_initialize_gpu(wavelet);

}

void wavelet_generate_fulltransformationmat(Wavelet *wavelet, int iteration, float **out_transf, int *out_numY, int *out_numX){
	//find number of details before approximation part
	int num_fixed_details = 0;
	for (int i=0; i < iteration; i++){
		num_fixed_details += wavelet->transform_numY[i]/2;
	}

	//dimensions of the transformation array
	int num_x = num_fixed_details + wavelet->transform_numX[iteration];
	int num_y = num_fixed_details + wavelet->transform_numY[iteration];
	float *transf = new float[num_x*num_y](); //initialize to zero

	//fill transformation matrix
	//identity matrix to keep previous details
	for (int row=0; row < num_fixed_details; row++){
		transf[row*num_x + row] = 1.0f;
	}

	//transformation matrix at the correct position for application to approximation part of array
	for (int row = num_fixed_details; row < num_y; row++){
		for (int col = num_fixed_details; col < num_x; col++){
			transf[row*num_x + col] = wavelet->transform_matrices[iteration][(row - num_fixed_details)*wavelet->transform_numX[iteration] + col - num_fixed_details];
		}
	}

	*out_transf = transf;
	*out_numY = num_y;
	*out_numX = num_x;
}

void wavelet_free(Wavelet *wavelet){
	for (int i=0; i < wavelet->transform_numIterations; i++){
		delete [] wavelet->transform_matrices[i];
	}
	delete [] wavelet->transform_matrices;
	delete [] wavelet->transform_numX;
	delete [] wavelet->transform_numY;
	delete [] wavelet->filter_h;
	delete [] wavelet->filter_g;
	delete [] wavelet->transform_fullmatrix;
}

void wavelet_run_transformation_filterbank(Wavelet *wavelet, float *dataLine, float *transformed){
	//run transform using simple filter bank algorithm
	float *approximation = new float[wavelet->image_samples*wavelet->transformed_length];
	memcpy(approximation, dataLine, sizeof(float)*wavelet->image_samples*wavelet->image_bands);
	int skipBands = 0;
	for (int iter = 0; iter < wavelet->transform_numIterations; iter++){
		//temporary output array
		float *temp_transformed = new float[wavelet->transform_numY[iter]*wavelet->image_samples]();

		//transform
		for (int i=0; i < wavelet->transform_numY[iter]/2; i++){
			for (int k=0; k < wavelet->filter_numCoeff; k++){
				for (int j=0; j < wavelet->image_samples; j++){
					int ind = 2*i+1 - k;
					if (ind < 0){
						ind = abs(ind+1);
					}
					if (ind >= wavelet->transform_numX[iter]){
						ind = 2*wavelet->transform_numX[iter] - ind - 1;
					}
					temp_transformed[(i + wavelet->transform_numY[iter]/2)*wavelet->image_samples + j] += approximation[ind*wavelet->image_samples + j]*wavelet->filter_h[k];
					temp_transformed[i*wavelet->image_samples + j] += approximation[ind*wavelet->image_samples + j]*wavelet->filter_g[k];
				}
			}
		}

		//copy transformed values to total transformation array
		memcpy(transformed + skipBands*wavelet->image_samples, temp_transformed, sizeof(float)*wavelet->transform_numY[iter]*wavelet->image_samples);

		//copy approximations to approximation array
		memcpy(approximation, temp_transformed + wavelet->image_samples*wavelet->transform_numY[iter]/2, sizeof(float)*wavelet->transform_numY[iter]/2*wavelet->image_samples);

		//update number of bands to skip
		skipBands += wavelet->transform_numY[iter]/2;

		delete [] temp_transformed;
	}

	delete [] approximation;
}

void wavelet_run_transformation_filterbank_matrix(Wavelet *wavelet, float *dataLine, float *transformed){
	//run transform using simple filter bank algorithm
	float *approximation = new float[wavelet->image_samples*wavelet->transformed_length];
	memcpy(approximation, dataLine, sizeof(float)*wavelet->image_samples*wavelet->image_bands);
	int skipBands = 0;

	const int NUM_FILTERS = 2;
	float *filterMat = new float[NUM_FILTERS*wavelet->filter_numCoeff];

	for (int i=0; i < wavelet->filter_numCoeff; i++){
		filterMat[i] = wavelet->filter_g[wavelet->filter_numCoeff - 1 - i];
		filterMat[wavelet->filter_numCoeff + i] = wavelet->filter_h[wavelet->filter_numCoeff - 1 - i];
	}

	for (int iter = 0; iter < wavelet->transform_numIterations; iter++){
		//temporary output array
		float *temp_transformed_approx = new float[wavelet->transform_numY[iter]*wavelet->image_samples]();
		float *temp_transformed_detail = new float[wavelet->transform_numY[iter]*wavelet->image_samples]();
		float *subMat = new float[wavelet->filter_numCoeff*wavelet->image_samples];

		//transform
		for (int i=0; i < wavelet->transform_numY[iter]/2; i++){
			int subMatBandInd = 0;

			//prepare multiplication matrix
			//start and end bands for filter convolution
			int startBand = 2*i+1-(wavelet->filter_numCoeff-1);
			int endBand = 2*i+1;

			//start and end bands for direct contigous memory copying
			int startContigCopy = startBand;
			int endContigCopy = endBand;

			//take care of boundary conditions
			//last filter coefficient will touch before the array boundaries
			if (startBand < 0){
				int startInd = startBand;
				int endInd = 0;
				for (int k=startInd; k < endInd; k++){
					int copyBandInd = abs(k+1);
					memcpy(subMat + subMatBandInd*wavelet->image_samples, approximation + copyBandInd*wavelet->image_samples, sizeof(float)*wavelet->image_samples);
					subMatBandInd++;
				}
				startContigCopy = endInd;
			}
	
			//middle segment
			endContigCopy = min(endContigCopy, wavelet->transform_numX[iter]-1);
			int numCopyBands = endContigCopy - startContigCopy + 1;
			memcpy(subMat + subMatBandInd*wavelet->image_samples, approximation + startContigCopy*wavelet->image_samples, sizeof(float)*wavelet->image_samples*numCopyBands);
			subMatBandInd += numCopyBands;
			
			//first filter coefficient will touch after the end of the array
			if (endBand > wavelet->transform_numX[iter]){
				int startInd = wavelet->transform_numX[iter];
				int endInd = endBand;
				for (int k = startInd; k <= endInd; k++){
					int copyBandInd = 2*wavelet->transform_numX[iter] - k - 1;
					memcpy(subMat + subMatBandInd*wavelet->image_samples, approximation + copyBandInd*wavelet->image_samples, sizeof(float)*wavelet->image_samples);
					subMatBandInd++;
				}
			}
			
			//calculate wavelet transform
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NUM_FILTERS, wavelet->image_samples, wavelet->filter_numCoeff, 1.0f, filterMat, wavelet->filter_numCoeff, subMat, wavelet->image_samples, 0.0f, temp_transformed_detail + i*wavelet->image_samples, wavelet->image_samples);

			//copy h part to the appropriate part of the transform matrix
			memcpy(temp_transformed_approx + i*wavelet->image_samples, temp_transformed_detail + (i+1)*wavelet->image_samples, sizeof(float)*wavelet->image_samples);
		}
		delete [] subMat;

		//copy transformed values to total transformation array
		memcpy(transformed + skipBands*wavelet->image_samples, temp_transformed_detail, sizeof(float)*wavelet->transform_numY[iter]/2*wavelet->image_samples);
		memcpy(transformed + (skipBands + wavelet->transform_numY[iter]/2)*wavelet->image_samples, temp_transformed_approx, sizeof(float)*wavelet->transform_numY[iter]/2*wavelet->image_samples);

		//copy approximations to approximation array
		memcpy(approximation, temp_transformed_approx, sizeof(float)*wavelet->transform_numY[iter]/2*wavelet->image_samples);

		//update number of bands to skip
		skipBands += wavelet->transform_numY[iter]/2;

		delete [] temp_transformed_detail;
		delete [] temp_transformed_approx;
	}
	delete [] approximation;
	delete [] filterMat;
}

void wavelet_run_transform(Wavelet *wavelet, float *dataLine, float **out_transformed, int *out_y, int *out_x, CalcType calcType){
	int skipBands = 0; //number of bands to skip in order to skip details part
	float *transformed = new float[wavelet->image_samples*wavelet->transformed_length];

	float *approximation = NULL;
	float *temp_transformed = NULL;

	switch (calcType){
		case USE_PARTIAL_MATRICES: 
			//use each individual transformation matrix, applied on approximation array
			approximation = new float[wavelet->image_samples*wavelet->transformed_length];
			memcpy(approximation, dataLine, sizeof(float)*wavelet->image_samples*wavelet->image_bands);
			
			//transformation
			for (int i=0; i < wavelet->transform_numIterations; i++){
				//temporary output array
				temp_transformed = new float[wavelet->transform_numY[i]*wavelet->image_samples];

				//transform
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, wavelet->transform_numY[i], wavelet->image_samples, wavelet->transform_numX[i], 1.0f, wavelet->transform_matrices[i], wavelet->transform_numX[i], approximation, wavelet->image_samples, 0.0f, temp_transformed, wavelet->image_samples);

				//copy transformed values to total transformation array
				memcpy(transformed + skipBands*wavelet->image_samples, temp_transformed, sizeof(float)*wavelet->transform_numY[i]*wavelet->image_samples);

				//copy approximations to approximation array
				memcpy(approximation, temp_transformed + wavelet->image_samples*wavelet->transform_numY[i]/2, sizeof(float)*wavelet->transform_numY[i]/2*wavelet->image_samples);

				//update number of bands to skip
				skipBands += wavelet->transform_numY[i]/2;

				delete [] temp_transformed;
			}
			delete [] approximation;
		break;
		case USE_FULL_MATRIX:
			//apply the full transformation matrix on the signal array, only one matrix multiplication and no memcpy
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, wavelet->transform_fullmatrix_numY, wavelet->image_samples, wavelet->transform_fullmatrix_numX, 1.0f, wavelet->transform_fullmatrix, wavelet->transform_fullmatrix_numX, dataLine, wavelet->image_samples, 0.0f, transformed, wavelet->image_samples);
		break;
		case USE_FILTER_BANK:
			wavelet_run_transformation_filterbank(wavelet, dataLine, transformed);
		break;
		case USE_FILTER_BANK_MATRIX:
			wavelet_run_transformation_filterbank_matrix(wavelet, dataLine, transformed);
		break;
		case USE_GPU:
			wavelet_run_transformation_gpu(wavelet, dataLine, transformed);
		break;
	}	

	//go through and keep only the specified indices
	if (wavelet->keep_numIndices == wavelet->transformed_length){
		//keep everything
		*out_transformed = transformed;
		*out_x = wavelet->image_samples;
		*out_y = wavelet->transformed_length;
	} else {
		float *out_temp = new float[wavelet->image_samples*wavelet->keep_numIndices];

		//go through indices to keep
		for (int i=0; i < wavelet->keep_numIndices; i++){
			memcpy(out_temp + i*wavelet->image_samples, transformed + wavelet->image_samples*wavelet->keep_indices[i], sizeof(float)*wavelet->image_samples);
		}
		delete [] transformed;
		*out_transformed = out_temp;
		*out_x = wavelet->image_samples;
		*out_y = wavelet->keep_numIndices;
	}
}


void wavelet_get_wavelet_function(Wavelet *wavelet, int row, float *wlet){
	memcpy(wlet, wavelet->transform_fullmatrix + wavelet->transform_fullmatrix_numX*row, sizeof(float)*wavelet->transform_fullmatrix_numX); 
}

void wavelet_get_wavelet_function(Wavelet *wavelet, int iteration, int detail, float *wlet){
	//calculate row in full transformation matrix
	int row = 0;
	for (int i=0; i < iteration; i++){
		row += wavelet->transform_numY[i]/2;
	}
	row += detail;
	wavelet_get_wavelet_function(wavelet, row, wlet);
}

void wavelet_get_wavelet_name(Wavelet *wavelet, int row, int *iteration, int *detail){
	*iteration = 0;
	*detail = 0;
	while (row >= 0){
		row -= wavelet->transform_numY[*iteration]/2;
		(*iteration)++;
	}
	(*iteration)--;
	row += wavelet->transform_numY[*iteration]/2;
	*detail = row;
}

int wavelet_get_band(Wavelet *wavelet, int iteration, int detail){
	int preliminary_row = 0;
	for (int i=0; i < iteration; i++){
		preliminary_row += wavelet->transform_numY[i]/2;
	}
	preliminary_row += detail;
	
	//look for row in keepindices:
	if (wavelet->keep_indices != NULL){
		for (int i=0; i < wavelet->keep_numIndices; i++){
			if (wavelet->keep_indices[i] == preliminary_row){
				return i;
			}
		}
	} else {
		return preliminary_row;
	}
	return -1;
}
