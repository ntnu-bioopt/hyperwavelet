//=======================================================================================================
// Copyright 2015 Asgeir Bjorgan, Lise Lyngsnes Randeberg, Norwegian University of Science and Technology
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
// http://opensource.org/licenses/MIT)
//=======================================================================================================

#include "readimage.h"
#include "wavelets.h"
#include <string.h>
#include <stdio.h>
using namespace std;

int main(int argc, char *argv[]){
	if (argc < 3){
		printf("Usage: %s input_file output_file\n", argv[0]);
		return 1;
	}

	//read image header
	char *filename = argv[1];
	size_t offset;
	HyspexHeader header;
	hyperspectral_read_header(filename, &header);

	//define image subset
	ImageSubset subset;
	subset.startSamp = 0;
	subset.endSamp = header.samples;
	subset.startLine = 0;
	subset.endLine = header.lines;
	int num_samples = subset.endSamp - subset.startSamp;
	int num_lines = subset.endLine - subset.startLine;
	
	int num_bands = header.bands;
	Wavelet wavelet;
	wavelet_initialize(&wavelet, SYM4, 6, num_bands, num_samples); //output all wavelet indices
	
	//output image
	int num_out_bands = wavelet.keep_numIndices;
	float *out_image = new float[num_out_bands*num_lines*num_samples];
	
	ImageSubset line_subset; //used for reading line by line, since the wavelet image can become quite large and we need to save some memory space. 
	line_subset.startSamp = 0;
	line_subset.endSamp = header.samples;

	for (int i=0; i < num_lines; i++){
		//read image line
		line_subset.startLine = i;
		line_subset.endLine = i+1;
		float *lineRefl = new float[num_samples*num_bands];
		hyperspectral_read_image(filename, &header, line_subset, lineRefl);
	
		//transform line
		float *transformedLine = NULL;
		int cols, rows;
		wavelet_run_transform(&wavelet, lineRefl, &transformedLine, &rows, &cols);
		memcpy(out_image + i*cols*rows, transformedLine, sizeof(float)*rows*cols);
		delete [] transformedLine;
		delete [] lineRefl;
	}

	//prepare image writing
	vector<float> wlens;
	for (int i=0; i < wavelet.keep_numIndices; i++){
		wlens.push_back(i);
	}

	//write output to hyperspectral file
	hyperspectral_write_header(argv[2], num_out_bands, num_samples, num_lines, wlens);
	hyperspectral_write_image(argv[2], num_out_bands, num_samples, num_lines, out_image);

	delete [] out_image;
	wavelet_free(&wavelet);
	return 0;
}
