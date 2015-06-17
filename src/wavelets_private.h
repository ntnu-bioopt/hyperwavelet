//HELPER FUNCTIONS, libhyperwavelet. 

#ifndef WAVELETS_PRIVATE_H_DEFINED
#define WAVELETS_PRIVATE_H_DEFINED

#include "wavelets.h"

//generate transformation matrix from input filter coefficients with symmetric boundary conditions
void wavelet_transform_matrix(float *filter_h, int filter_numCoeffs, int image_bands, float **out_transf, int *out_numY, int *out_numX);

//generate total transformation matrix from the two filter coefficients contained in wavelet object, for an input signal of length signal_length
void wavelet_generate_transformation(Wavelet *wavelet, int signal_length, float **out_transf, int *out_numY, int *out_numX);

//generate full transformation matrix when it would be applied to the full details/approximation-array and not only the approximation part
void wavelet_generate_fulltransformationmat(Wavelet *wavelet, int iteration, float **out_transf, int *out_numY, int *out_numX); 

//actually, all of this transform matrix stuff is kind of bullshit anyway since that in reality is the mother wavelet multiplied by the signal... And has a larger runtime than the filter bank algorithm. But due to internal banking and memory stuff in MKL, it still is faster than the for loops, though the asymptotic time is larger.   

//implementation using filter banks
void wavelet_run_transformation_filterbank(Wavelet *wavelet, float *dataLine, float *transformed);

//GPU functions 
void wavelet_initialize_gpu(Wavelet *wavelet);
void wavelet_run_transformation_gpu(Wavelet *wavelet, float *dataLine, float *out_transformed);

//symlet-4 wavelet properties
const int SYM4_NUMCOEFF = 8;
const float SYM4_H[SYM4_NUMCOEFF] = {-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427};
const float SYM4_G[SYM4_NUMCOEFF] = {-0.0322231006040427, -0.012603967262037833, 0.09921954357684722, 0.29785779560527736, -0.8037387518059161, 0.49761866763201545, 0.02963552764599851, -0.07576571478927333};


#endif
