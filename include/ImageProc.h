/*///////////////////////////////
# EE569 Homework Assignment #2
# Date: Oct 19, 2014
# Name: Rishabh Battulwar
# ID: 4438-1435-20
# email: battulwa@usc.edu
#
# Compiled on CYGWIN with g++
*/////////////////////////////////

#ifndef __IMAGE_PROC_H__
#define __IMAGE_PROC_H__

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <limits>
#include <string>
#include "Image.h"
#include "colorspace.h"

#define LEFT2RIGHT 1
#define RIGHT2LEFT 0

#define CMYW 1
#define MYGC 2
#define RGMY 3
#define CMBG 4
#define RGBM 5
#define KRGB 6

#define SHRINK_FIRST_PATTERNS 62
#define THIN_FIRST_PATTERNS 46
#define SKELETON_FIRST_PATTERNS 40
#define SHRINK_THIN_SECOND_PATTERNS 49
#define SKELETON_SECOND_PATTERNS 36

std::string shrink_first[SHRINK_FIRST_PATTERNS] = {	"11101111", "11111101", "11110111",	"10111111", //10stk
							"11101011",	"01101111", "11111100", "11111001", "11110110", "10011111", "00111111", //9stk
							"01101011", "11111000", "11010110", "00011111", //8stk
							"11101001", "11110100", "10010111", "00101111", //7stk
							"11101000", "01101001", "11110000", "11010100", "10010110", "00010111", "00101011", //6stk
							"11001001", "01110100", //6st
							"01101000", "11010000", "00010110", "00001011", //5st
							"11001000", "01001001", "01110000", "00101010", //5st
							"00101001", "11100000", "10010100", "00000111", //4stk
							"00101000", "01100000", "11000000", "10010000", "00010100", "00000110", "00000011", "00001001", //3s
							"00001000", "01000000", "00010000", "00000010", //2s
							"00100000", "10000000", "00000100", "00000001" //1s
						};

std::string thin_first[THIN_FIRST_PATTERNS] = { 	"11101111", "11111101", "11110111",	"10111111", //10stk
						"11101011",	"01101111", "11111100", "11111001", "11110110", "10011111", "00111111", //9stk
						"01101011", "11111000", "11010110", "00011111", //8stk
						"11101001", "11110100", "10010111", "00101111", //7stk
						"11101000", "01101001", "11110000", "11010100", "10010110", "00010111", "00101011", //6stk
						"11001001", "01110100", //6st
						"01101000", "11010000", "00010110", "00001011", //5st
						"11001000", "01001001", "01110000", "00101010", //5st
						"00101001", "11100000", "10010100", "00000111", //4stk
						"01001000", "01010000", "00010010", "00001010" //4tk
					};

std::string skeleton_first[SKELETON_FIRST_PATTERNS] = 	{
								"11111011", "1111110", "11011111", "01111111", //11k
								"11101111", "11111101", "11110111",	"10111111", //10stk
								"11101011",	"01101111", "11111100", "11111001", "11110110", "10011111", "00111111", //9stk
								"01101011", "11111000", "11010110", "00011111", //8stk
								"11101001", "11110100", "10010111", "00101111", //7stk
								"11101000", "01101001", "11110000", "11010100", "10010110", "00010111", "00101011", //6stk
								"00101001", "11100000", "10010100", "00000111", //4stk
								"01001000", "01010000", "00010010", "00001010" //4tk
							};

std::string shrink__thin_second[SHRINK_THIN_SECOND_PATTERNS] = {
									"00M00000", "M0000000", //Spur
									"000000M0", "0000M000", //Single 4-connection
									"00M0M000", "0MM00000", "MM000000", "M00M0000", //L cluster 1
									"000M0M00", "00000MM0", "000000MM", "0000M00M", //L cluster 2
									"0MMM0000", "MM00M000", "0M00M00M", "00M0M0M0", //4-connected offset
									"0MM00M00", "00M0MM00", "M00M000M", "MM00000M", "00MM0M00", "00M00MM0", "M00000MM", "M000M00M", //Spur corner cluster
									"MMDMDDDD", //Corner Cluster
									"DM0MMD00", "0MDMM00D", "00DMM0MD", "D00MMDM0", //Tee Branch 1
									"DMDM00M0", "0M0M0DMD", "0M00MDMD", "DMD0M0M0", //Tee Branch 2
									"MDMDD100", "MDMDD010", "MDMDD001", "MD0D0MD1", "MD0D1MD0", "MD1D0MD0", "001DDMDM", "010DDMDM", "100DDMDM", "1DM0D0DM", "0DM1D0DM", "0DM0D1DM", //Vee Branch
									"DM00MM0D", "0MDM0D0M", "D0MM00MD", "M0D0MDM0"
								};

std::string skeleton_second[SKELETON_SECOND_PATTERNS] = {
									"0000000M", "00000M00", "00M00000", "M0000000", //Spur
									"000000M0", "0000M000", "000M0000", "0M000000", //single 4-connection
									"0M00M000", "0M0M0000", "0000M0M0", "000M00M0", //L corner
									"MMDMDDDD", "DDDDMDMM", //Corner Cluster
									"DMDMMDDD", "DMDMDDMD", "DDDMMDMD", "DMDDMDMD", //Tee Branch
									"MDMDD100", "MDMDD010", "MDMDD001", "MD0D0MD1", "MD0D1MD0", "MD1D0MD0", "001DDMDM", "010DDMDM", "100DDMDM", "1DM0D0DM", "0DM1D0DM", "0DM0D1DM", //Vee Branch
									"DM00MM0D", "0MDM0D0M", "D0MM00MD", "M0D0MDM0"
								};
//#############################################################################################
//################################# CLASS DEFINITION begin ####################################
//#############################################################################################


class ImageProc
{
	private:

	public:
    	static void color_to_bw(const char *infile, const char *outfile, int width, int height);
		static void image_resize(const char *infile, const char* outfile, int width, int height, int target_width, int target_height);
		static void hist_equal_cumulative(const char *infile, const char *outfile, int width, int height);
		static void hist_equal(const char *infile, const char *outfile, int width, int height);
		static void oil_painting(const char *infile, const char *outfile, int width, int height, int window_size);
		static void oil_paint_filter(Image &img_orig, Image &img_modified, int window_size);
		static void film_special_effect(const char *infile, const char *outfile, int width, int height);
		static void film_effect_filter(Image &img_orig, Image &img_modified);
		static void denoising(const char *infile, const char *outfile, int width, int height, int window_size, int sigma, float threshold);
		static void gaussian_filter(Image &img_orig, Image &img_modified, int window_size, float sigma);
		static void gaussian_kernel(float **kernel, int window_size, int sigma);
		static void non_uniform_box_filter(Image &img_orig, Image &img_modified, int window_size);
		static void outlier_filter(Image &img_orig, Image &img_modified, int window_size, float threshold);
		static void median_filter(Image &img_orig, Image &img_modified, int window_size);
		static void apply_bilateral_filter(const char *infile, const char *outfile, int width, int height, int window_size, int sigma, int sigma_simi, int iter);
		static void bilateral_filter(Image &img_orig, Image &img_modified, int window_size, int sigma_dist, int sigma_simi);
		static void apply_non_local_mean(const char *infile, const char *outfile, int width, int height, int window_size, int region_size, int sigma, int h);
		static void non_local_mean(Image &img_orig, Image &img_modified, int window_size, int region_size, int sigma, int h);
		static void hist_equal_proc(Image &img_orig, Image &img_modified);
		static float get_PSNR(Image &img_orig, Image &img_modified);
		static float get_MSE(Image &img_orig, Image &img_modified);
		static void mergesort(int *a, int low, int high);
		static void merge(int *a, int low, int high, int mid);
		static int get_median(int *sorted_array, int size);
		static void get_histogram(Image &img, int histogram_bins[256]);
		static void get_cumulative_histogram(int histogram_bins[256], int hist_cumulative[256]);

		static void apply_sobel_operator(const char *infile, const char *outfile, int width, int height, int percentage);
		static void sobel_edge_detector(Image &img_orig, Image &img_modified, int percentage);
		static int func_gradient(int **operator_mask, Image &img_orig, int x, int y, int z, int window_size);
		static float func_gradient_float(float **operator_mask, Image &img_orig, int x, int y, int z, int window_size);
		static void apply_LoG_operator(const char *infile, const char *outfile, int width, int height);
		static void LoG_edge_detector(Image &img_orig, Image &img_modified);
		static int check_zero_crossing(Image &img_modified_copy, int i, int j, int k);
		static void halftoning(const char *infile, int width, int height);
		static void halftone_fixed_thresh(Image &img_orig, Image &img_modified);
		static void halftone_random_thresh(Image &img_orig, Image &img_modified);
		static void halftone_dither_matrix(Image &img_orig, Image &img_modified, const int matrix_size);
		static void error_diffusion(Image &img_orig, Image &img_modified);
		static void color_halftoning(const char *infile, const char *outfile, int width, int height);
		static void mbvq_error_diffusion(Image &img_orig, Image &img_modified);
		static float distance_from_vertex(float r, float g, float b, char vertex);
		static int nearest_vertex(float distance[8], int i, int j, int k, int l);
		static void apply_artistic_effect(const char *infile, const char *outfile, int width, int height, float tau, float epsilon, float phi, int window_size, int sigma_dist, int sigma_simi, int iter);
		static void rgb2cielab(Image &img_orig, Image &img_modified);
		static void cielab2rgb(Image &img_orig, Image &img_modified);
		static void apply_morphology(const char *infile, const char *outfile, int width, int height);
		static void unsharp_mask(Image &img_orig, Image &img_modified);
		static void otsu_th(Image &img_orig, Image &img_binary);
		static void shrink(Image &img_binary, Image &img_shrinked);
		static void thin(Image &img_binary, Image &img_thinned);
		static void skeletonize(Image &img_binary, Image &img_skeletonized);
		static void apply_binarization(const char *infile, const char *outfile, int width, int height);
		static void apply_minutae_extraction(const char* infile, const char *outfile, int width, int height);
		static void minutae_extraction(Image &img_orig, Image &img_modified);
};

//#############################################################################################
//################################# CLASS DEFINITION end ######################################
//#############################################################################################



//#############################################################################################
//########## Color to Black and White method ##################################################

void ImageProc::color_to_bw(const char *infile, const char *outfile, int width, int height)
{
	Image img_color("color", width, height);	//object for input color image
	Image img_bw("bw", width, height);			//object for output b-&-w image

	img_color.read_image(infile, img_color.cols, img_color.rows);	//reading input image from file

	for(int i = 0; i < img_bw.rows; i++)
		for(int j = 0; j < img_bw.cols; j++)
			for(int k = 0; k < img_bw.channels; k++)
				img_bw.setvalue(i,j,k,((img_color.getvalue(i,j,0)*0.21)\
									  +(img_color.getvalue(i,j,1)*0.72)\
									  +(img_color.getvalue(i,j,2)*0.07))); //using Luminosity formula: grayscale_value = 0.21*R + 0.72*G + 0.07*B

	img_bw.write_image(outfile, img_bw.cols, img_bw.rows);	//writing image to output file
}

//#############################################################################################
//#############################################################################################


//#############################################################################################
//########## Image Resizing method ############################################################

void ImageProc::image_resize(const char *infile, const char* outfile, int width, int height, int target_width, int target_height)
{
	Image img_orig("color", width, height);
	Image img_resized("color", target_width, target_height);

	img_orig.read_image(infile, img_orig.cols, img_orig.rows);

	float X, Y, x, y, a, b;

	for(int i = 0; i < img_resized.rows; i++)
		for(int j = 0; j < img_resized.cols; j++)
		{
			x = img_orig.rows * (1.0 * i / img_resized.rows);	//Scaling in x direction
			y = img_orig.cols * (1.0 * j / img_resized.cols);	//Scaling in x direction

			X = floor(x);
			Y = floor(y);

			a = y - Y;
			b = x - X;

			for(int k = 0; k < img_resized.channels; k++)
				img_resized.setvalue(i,j,k,((img_orig.getvalue(X  ,Y  ,k)*(1.0-a)*(1.0-b))
				  						   +(img_orig.getvalue(X  ,Y+1,k)*(    a)*(1.0-b))
				  						   +(img_orig.getvalue(X+1,Y  ,k)*(1.0-a)*(    b))
				  						   +(img_orig.getvalue(X+1,Y+1,k)*(    a)*(    b))		//using 4 surrounding co-ordinates
				  						   )
									);															//BILINEAR INTERPOLATION FORMULATION
		}

	img_resized.write_image(outfile, img_resized.cols, img_resized.rows);
}

//#############################################################################################
//#############################################################################################


//#############################################################################################
//########### Histogram Equalization using CDF :: going to majority ###########################

void ImageProc::hist_equal_cumulative(const char *infile, const char *outfile, int width, int height)
{
	//########### If image == bw #########
	// Image img_bw("bw", width, height);

	// img_bw.read_image(infile, width, height);

	// int histogram_bins[256] = {0};

	// for(int i = 0; i < height; i++)
	// 	for(int j = 0; j < width; j++)
	// 		if((img_bw(i, j, 0) >= 0) && (img_bw(i, j, 0) < 256))
	// 			histogram_bins[img_bw(i, j, 0)]++;


	// int hist_cumulative[256] = {0};
	// hist_cumulative[0] = histogram_bins[0];
	// for(int i = 1; i < 256; i++)
	// 	hist_cumulative[i] = hist_cumulative[i-1] + histogram_bins[i];


	// for(int i = 0; i < height; i++)
	// 	for(int j = 0; j < width; j++)
	// 		if      (img_bw(i, j, 0) < 0  ) img_bw.imgdata[(i)*width*1 + (j)*1 + 0] = 0  ;
	// 		else if (img_bw(i, j, 0) > 255) img_bw.imgdata[(i)*width*1 + (j)*1 + 0] = 255;
	// 		else    img_bw.imgdata[(i)*width*1 + (j)*1 + 0] = 255 * hist_cumulative[img_bw(i,j,0)] / hist_cumulative[255];


	// img_bw.write_image(outfile, width, height);
	//###################################


	//######## If image == color ########
	Image img_color("color", width, height);

	img_color.read_image(infile, width, height);

	int histogram_bins[256][3] = {0};

	for(int i = 0; i < img_color.rows; i++)
		for(int j = 0; j < img_color.cols; j++)
			for(int k = 0; k < img_color.channels; k++)
				if((img_color.getvalue(i,j,k) >= 0) && (img_color.getvalue(i,j,k) < 256))
					histogram_bins[img_color.getvalue(i,j,k)][k]++;


	int hist_cumulative[256][3] = {0};
	for(int k = 0; k < 3; k++)
		hist_cumulative[0][k] = histogram_bins[0][k];
	for(int i = 1; i < 256; i++)
		for(int k = 0; k < 3; k++)
			hist_cumulative[i][k] = hist_cumulative[i-1][k] + histogram_bins[i][k];

	// std::ofstream file;

	// file.open("rt.txt");
	// for(int i = 0; i < 256; i++)
	// 	file << i << "\t" <<hist_cumulative[i][0] << "\n";
	// file.close();

	// file.open("gt.txt");
	// for(int i = 0; i < 256; i++)
	// 	file << i << "\t" <<hist_cumulative[i][1] << "\n";
	// file.close();

	// file.open("bt.txt");
	// for(int i = 0; i < 256; i++)
	// 	file << i << "\t" <<hist_cumulative[i][2] << "\n";
	// file.close();


	for(int i = 0; i < height; i++)
		for(int j = 0; j < width; j++)
			for(int k = 0; k < 3; k++)
				if      (img_color.getvalue(i,j,k) < 0  ) img_color.setvalue(i,j,k,0)  ;
				else if (img_color.getvalue(i,j,k) > 255) img_color.setvalue(i,j,k,255);
				else    img_color.setvalue(i,j,k,(255 * hist_cumulative[img_color.getvalue(i,j,k)][k] / hist_cumulative[255][k]));


	img_color.write_image(outfile, width, height);
	//###################################
}

//#############################################################################################
//#############################################################################################


//#############################################################################################
//########### Histogram Equalization using CDF ################################################

void ImageProc::hist_equal(const char *infile, const char *outfile, int width, int height)
{
	//########### If image == bw #########
	// Image img_bw("bw", width, height);

	// img_bw.read_image(infile, width, height);

	// int histogram_bins[256] = {0};

	// for(int i = 0; i < height; i++)
	// 	for(int j = 0; j < width; j++)
	// 		if((img_bw(i, j, 0) >= 0) && (img_bw(i, j, 0) < 256))
	// 			histogram_bins[img_bw(i, j, 0)]++;


	// int hist_cumulative[256] = {0};
	// hist_cumulative[0] = histogram_bins[0];
	// for(int i = 1; i < 256; i++)
	// 	hist_cumulative[i] = hist_cumulative[i-1] + histogram_bins[i];

	// int hist_cumulative_copy[256] = {0};
	// int curr_bin_count = 0, prev_bin_count = 0;
	// double value =  0;
	// for(int i = 0; i < height; i++)
	// 	for(int j = 0; j < width; j++)
	// 		if      (img_bw(i, j, 0) < 0  ) img_bw.imgdata[(i)*width*1 + (j)*1 + 0] = 0  ;
	// 		else if (img_bw(i, j, 0) > 255) img_bw.imgdata[(i)*width*1 + (j)*1 + 0] = 255;
	// 		else
	// 		{
	// 			++hist_cumulative_copy[img_bw(i, j, 0)];
	// 			curr_bin_count = hist_cumulative_copy[img_bw(i, j, 0)];
	// 			prev_bin_count = hist_cumulative[img_bw(i, j, 0) - 1];
	// 			value = floor(255.0 * (prev_bin_count + curr_bin_count) / hist_cumulative[255]);
	// 			img_bw.imgdata[(i)*width*1 + (j)*1 + 0] = value;
	// 			value = 0; curr_bin_count = 0; prev_bin_count = 0;
	// 		}

	// img_bw.write_image(outfile, width, height);
	//###################################


	//######## If image == color ########
	Image img_color("color", width, height);

	img_color.read_image(infile, width, height);

	int histogram_bins[256][3] = {0};

	for(int i = 0; i < img_color.rows; i++)
		for(int j = 0; j < img_color.cols; j++)
			for(int k = 0; k < img_color.channels; k++)
				if((img_color(i, j, k) >= 0) && (img_color(i, j, k) < 256))
					histogram_bins[img_color(i, j, k)][k]++;


	int hist_cumulative[256][3] = {0};
	for(int k = 0; k < 3; k++)
		hist_cumulative[0][k] = histogram_bins[0][k];
	for(int i = 1; i < 256; i++)
		for(int k = 0; k < 3; k++)
			hist_cumulative[i][k] = hist_cumulative[i-1][k] + histogram_bins[i][k];

	// std::ofstream file;

	// file.open("rt2.txt");
	// for(int i = 0; i < 256; i++)
	// 	file << i << "\t" <<hist_cumulative[i][0] << "\n";
	// file.close();

	// file.open("gt2.txt");
	// for(int i = 0; i < 256; i++)
	// 	file << i << "\t" <<hist_cumulative[i][1] << "\n";
	// file.close();

	// file.open("bt2.txt");
	// for(int i = 0; i < 256; i++)
	// 	file << i << "\t" <<hist_cumulative[i][2] << "\n";
	// file.close();

	int hist_cumulative_copy[256][3] = {0};
	int curr_bin_count = 0, prev_bin_count = 0;
	double value =  0;

	for(int i = 0; i < img_color.rows; i++)
		for(int j = 0; j < img_color.cols; j++)
			for(int k = 0; k < 3; k++)
				if      (img_color.getvalue(i,j,k) < 0  ) img_color.setvalue(i,j,k,0);
				else if (img_color.getvalue(i,j,k) > 255) img_color.setvalue(i,j,k,255);
				else
				{
					++hist_cumulative_copy[img_color(i, j, k)][k];
					curr_bin_count = hist_cumulative_copy[img_color(i, j, k)][k];
					prev_bin_count = hist_cumulative[img_color(i, j, k) - 1][k];
					value = floor(255.0 * (prev_bin_count + curr_bin_count) / hist_cumulative[255][k]);
					img_color.setvalue(i,j,k,value);
					value = 0; curr_bin_count = 0; prev_bin_count = 0;
				}


	img_color.write_image(outfile, img_color.cols, img_color.rows);
	//###################################
}

//#############################################################################################
//#############################################################################################


//#############################################################################################
//########### Oil Painting Effect #############################################################

void ImageProc::oil_painting(const char *infile, const char *outfile, int width, int height, int window_size)
{
	Image img_orig("color", width, height);
	Image img_oil("color", width, height);

	img_orig.read_image(infile, img_orig.cols, img_orig.rows);
	oil_paint_filter(img_orig, img_oil, window_size);

	img_oil.write_image(outfile, img_oil.cols, img_oil.rows);
}

//#############################################################################################

void ImageProc::oil_paint_filter(Image &img_orig, Image &img_modified, int window_size)
{
	int ws = window_size; //window_size
	int wleft = ((-1*ws) + 1) / 2;
	int wright = (( 1*ws) - 1) / 2;
	int values[ws*ws][img_orig.channels];
	int max_occur_val[img_orig.channels], max_occurrence = 0, occurrence = 0;;
	int index;

	for(int i = 0; i < img_orig.rows; i++)
	 	for(int j = 0; j < img_orig.cols; j++)
	 	{
			max_occur_val[0] = max_occur_val[1] = max_occur_val[2] = 0;
			max_occurrence = 0;
			for(index = 0; index < ws*ws; index++)
				values[index][0] = values[index][1] = values[index][2] = 0;
			index = 0;

			for(int m = wleft; m <= wright; m++)
				for(int n = wleft; n <= wright; n++)
				{
					if((i+m >= 0) && (i+m < img_orig.rows) && (j+n >= 0) && (j+n < img_orig.cols))
						for(int k = 0; k < img_orig.channels; k++)
						{
							values[index][k] = img_orig.getvalue(i+m, j+n, k);
						}
					else
						for(int k = 0; k < img_orig.channels; k++)
							values[index][k] = 280;

					index++;

				}


//############ FIXXX for img_orig.channels
			for(index = 0; index < ws*ws; index++)
			{
				if(!(values[index][0] == 280) && !(values[index][1] == 280) && !(values[index][2] == 280))
				{
					occurrence = 0;
					for(int r = 0; r < ws*ws; r++)
					{
						if(values[index][0] == values[r][0] &&
						   values[index][1] == values[r][1] &&
						   values[index][2] == values[r][2])
							occurrence++;

					}

					if(max_occurrence < occurrence)
					{
						max_occurrence = occurrence;
						max_occur_val[0] = values[index][0];
						max_occur_val[1] = values[index][1];
						max_occur_val[2] = values[index][2];
					}
				}
			}
//#############

			img_modified.setvalue(i,j,0,max_occur_val[0]);
			img_modified.setvalue(i,j,1,max_occur_val[1]);
			img_modified.setvalue(i,j,2,max_occur_val[2]);


	 	}
}

//#############################################################################################
//#############################################################################################


//#############################################################################################
//########### Film Special Effect #############################################################

void ImageProc::film_special_effect(const char *infile, const char *outfile, int width, int height)
{
	Image img_orig("color", width, height);
	Image img_film("color", width, height);

	img_orig.read_image(infile, img_orig.cols, img_orig.rows);
	film_effect_filter(img_orig, img_film);

	img_film.write_image(outfile, img_film.cols, img_film.rows);
}

//#############################################################################################

void ImageProc::film_effect_filter(Image &img_orig, Image &img_modified)
{
	for(int i = 0; i < img_modified.rows; i++)
		for(int j = 0; j < img_modified.cols; j++)
			for(int k = 0; k < img_modified.channels; k++)
			{
				img_modified.setvalue(i,j,k,(255-img_orig.getvalue(i,(img_orig.cols-j),k)));
			}

	int gmax[3] = {255,205,180};
	int gmin[3] = { 80, 30, 20};
	float scaling_factor[3];
	for(int i = 0; i < 3; i++)
		scaling_factor[i] = (gmax[i]-gmin[i]) / 255.0;
	for(int i = 0; i < img_modified.rows; i++)
		for(int j = 0; j < img_modified.cols; j++)
			for(int k = 0; k < img_modified.channels; k++)
			{
				img_modified.setvalue(i,j,k,((img_modified.getvalue(i,j,k)
											  * scaling_factor[k])
											  + gmin[k]));
			}
}

//#############################################################################################
//#############################################################################################


//#############################################################################################
//########### Denoising #######################################################################

void ImageProc::denoising(const char *infile, const char *outfile, int width, int height, int window_size, int sigma, float threshold)
{
	Image img_orig("color", width, height);
	Image img_result("color", width, height);
	Image img_Lena("color", width, height);

	img_orig.read_image(infile, img_orig.cols, img_orig.rows);
	img_Lena.read_image("Lena.raw", img_Lena.cols, img_Lena.rows);

	//Separate channels
	Image img_red("bw", img_orig.cols, img_orig.rows);
	Image img_blue("bw", img_orig.cols, img_orig.rows);
	Image img_green("bw", img_orig.cols, img_orig.rows);
	Image img_redOut("bw", img_orig.cols, img_orig.rows);
	Image img_greenOut("bw", img_orig.cols, img_orig.rows);
	Image img_blueOut("bw", img_orig.cols, img_orig.rows);

	Image img_LenaR("bw", img_Lena.cols, img_Lena.rows);
	Image img_LenaB("bw", img_Lena.cols, img_Lena.rows);
	Image img_LenaG("bw", img_Lena.cols, img_Lena.rows);

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			img_red.setvalue(i,j,0,img_orig.getvalue(i,j,0));
			img_green.setvalue(i,j,0,img_orig.getvalue(i,j,1));
			img_blue.setvalue(i,j,0,img_orig.getvalue(i,j,2));
		}

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			img_LenaR.setvalue(i,j,0,img_Lena.getvalue(i,j,0));
			img_LenaG.setvalue(i,j,0,img_Lena.getvalue(i,j,1));
			img_LenaB.setvalue(i,j,0,img_Lena.getvalue(i,j,2));
		}

	median_filter(img_red, img_redOut, window_size);
	median_filter(img_green, img_greenOut, window_size);
	median_filter(img_blue, img_blueOut, window_size);

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			img_red.setvalue(i,j,0,img_redOut.getvalue(i,j,0));
			//img_green.setvalue(i,j,0,img_greenOut.getvalue(i,j,0));
			img_blue.setvalue(i,j,0,img_blueOut.getvalue(i,j,0));
		}

	gaussian_filter(img_red, img_redOut, window_size, sigma);
	//gaussian_filter(img_green, img_greenOut, window_size, sigma);
	gaussian_filter(img_blue, img_blueOut, window_size, sigma);

	float PSNR_red   = get_PSNR(img_red, img_redOut);
	float PSNR_green = get_PSNR(img_green, img_greenOut);
	float PSNR_blue  = get_PSNR(img_blue, img_blueOut);

	float PSNR_LenaR   = get_PSNR(img_LenaR, img_redOut);
	float PSNR_LenaG   = get_PSNR(img_LenaG, img_greenOut);
	float PSNR_LenaB   = get_PSNR(img_LenaB, img_blueOut);

	std::cout << "PSNR_red : " << PSNR_red << "\nPSNR_green : " << PSNR_green\
			<< "\nPSNR_blue : " << PSNR_blue << "\nPSNR_LenaR : " << PSNR_LenaR << "\nPSNR_LenaG: " << PSNR_LenaG\
			<< "\nPSNR_LenaB : " << PSNR_LenaB << std::endl;
	// hist_equal_proc(img_red, img_red_O);
	// hist_equal_proc(img_blue, img_blue_O);
	// hist_equal_proc(img_green, img_green_O);

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			img_result.setvalue(i,j,0,img_redOut.getvalue(i,j,0));
			img_result.setvalue(i,j,1,img_greenOut.getvalue(i,j,0));
			img_result.setvalue(i,j,2,img_blueOut.getvalue(i,j,0));
		}

	img_result.write_image(outfile, img_result.cols, img_result.rows);

}

//#############################################################################################

void ImageProc::gaussian_filter(Image &img_orig, Image &img_modified, int window_size, float sigma)
{
	int ws = window_size;
	float kernel[ws][ws]; //discrete convolution operator

	//gaussian_kernel(kernel, ws, sigma);
	float kernel_sum = 0.0;
    int kernel_radius = ws / 2;
    //double coeff = 1.0 / (2.0 * M_PI * (sigma * sigma));
    float frac = 0.0;

    for (int row = -kernel_radius; row <= kernel_radius; row++)
        for (int col = -kernel_radius; col <= kernel_radius; col++)
        {
            frac = exp(-1.0 * ((row*row) + (col*col)) / (2.0 * (sigma*sigma)));

            kernel[row + kernel_radius][col + kernel_radius] = frac;

            kernel_sum += kernel[row + kernel_radius][col + kernel_radius];
        }

    for (int i = 0; i < ws; i++)
        for (int j = 0; j < ws; j++)
            kernel[i][j] = kernel[i][j] / kernel_sum; //Normalized


	for(int i = 0; i < img_orig.rows; i++)
	 	for(int j = 0; j < img_orig.cols; j++)
	 		for(int k = 0; k < img_orig.channels; k++)
		 	{
				float values[ws][ws];
				for(int row = 0; row < ws; row++)
					for(int col = 0; col < ws; col++)
						values[row][col] = 0;
				float values_avg = 0;

				for(int m = -kernel_radius, row = 0; m <= kernel_radius; m++, row++)
					for(int n = -kernel_radius, col = 0; n <= kernel_radius; n++, col++)
					{
						if((i+m >= 0) && (i+m < img_orig.rows) && (j+n >= 0) && (j+n < img_orig.cols))
						{
							values[row][col] = 1.0 * img_orig.getvalue((i+m),(j+n),k) * kernel[row][col];
						}
						else
						{
							values[row][col] = 1.0 * img_orig.getvalue((i+m),(j-n),k) * kernel[row][col];
						}
					}

				for(int row = 0; row < ws; row++)
					for(int col = 0; col < ws; col++)
						values_avg += values[row][col];

				img_modified.setvalue(i,j,k,(int)values_avg);
			}

}

//#############################################################################################

void ImageProc::non_uniform_box_filter(Image &img_orig, Image &img_modified, int window_size)
{
	int ws = window_size; //window_size
	int wleft = ((-1*ws) + 1) / 2;
	int wright = (( 1*ws) - 1) / 2;
	int wtop = wleft;
	int wbottom = wright;

	double filter_kernel[ws][ws], kernel_sum = 0, normalized_kernel[ws][ws];
	double row_exp = 0, col_exp;
	for(int i = wtop, row = 0; i <= wbottom; i++, row++)
	{
		col_exp = row_exp;

		for(int j = wleft, col = 0; j <= wright; j++, col++)
		{
			filter_kernel[row][col] = pow(2.0, col_exp);

			if(j < 0)
				col_exp++;
			if(j >= 0)
				col_exp--;
		}

		if(i < 0)
			row_exp++;

		if(i >= 0)
			row_exp--;
	}

	for(int row = 0; row < ws; row++)
		for(int col = 0; col < ws; col++)
			kernel_sum += filter_kernel[row][col];

	for(int row = 0; row < ws; row++)
		for(int col = 0; col < ws; col++)
			normalized_kernel[row][col] = filter_kernel[row][col] / kernel_sum;

	for(int i = 0; i < img_orig.rows; i++)
	 	for(int j = 0; j < img_orig.cols; j++)
	 		for(int k = 0; k < img_orig.channels; k++)
		 	{
				int values[ws][ws];
				for(int row = 0; row < ws; row++)
					for(int col = 0; col < ws; col++)
						values[row][col] = 0;
				int values_avg = 0;

				for(int m = wleft, row = 0; m <= wright; m++, row++)
					for(int n = wtop, col = 0; n <= wbottom; n++, col++)
					{
						if((i+m >= 0) && (i+m < img_orig.rows) && (j+n >= 0) && (j+n < img_orig.cols))
						{
							values[row][col] = img_orig.getvalue((i+m),(j+n),k) * normalized_kernel[row][col];
						}
						else
						{
							values[row][col] = img_orig.getvalue((i+m),(j-n),k) * normalized_kernel[row][col];
						}
					}

				for(int row = 0; row < ws; row++)
					for(int col = 0; col < ws; col++)
						values_avg += values[row][col];

				img_modified.setvalue(i,j,k,values_avg);
			}

}

//#############################################################################################

void ImageProc::outlier_filter(Image &img_orig, Image &img_modified, int window_size, float threshold)
{
	int ws = window_size;
	int kernel_radius = ws / 2;
	int values[ws][ws];
	int values_sum = 0;
	int new_value = 0;
	int diff;

	for(int i = 0; i < img_orig.rows; i++)
	 	for(int j = 0; j < img_orig.cols; j++)
	 		for(int k = 0; k < img_orig.channels; k++)
		 	{
				for(int row = 0; row < ws; row++)
					for(int col = 0; col < ws; col++)
						values[row][col] = 0;

				values_sum = 0;
				new_value = 0;
				diff = 0.0;

				for(int m = -kernel_radius, row = 0; m <= kernel_radius; m++, row++)
					for(int n = -kernel_radius, col = 0; n <= kernel_radius; n++, col++)
					{
						if((i+m >= 0) && (i+m < img_orig.rows) && (j+n >= 0) && (j+n < img_orig.cols))
						{
							values[row][col] = img_orig.getvalue((i+m),(j+n),k);
							values_sum += values[row][col];
							//std::cout << values[row][col] << std::endl;

						}
						else
						{
							values[row][col] = img_orig.getvalue((i+m),(j-n),k);
							values_sum += values[row][col];
							//std::cout << values[row][col] << std::endl;
						}
					}

				//std::cout << values_sum << std::endl;

				diff = abs((img_orig.getvalue(i,j,k)) - ((1.0/((ws*ws)-1)) * (values_sum - img_orig.getvalue(i,j,k))));
				//std::cout << "diff : " << diff << std::endl;
				//std::cout << "gt : " << (img_orig.getvalue(i,j,k)) << " " << ((1.0/((ws*ws)-1)) * (values_sum - img_orig.getvalue(i,j,k))) << std::endl;
				if(diff > threshold)
				{
					new_value = (1.0/((ws*ws)-1)) * (values_sum - img_orig.getvalue(i,j,k));
					//std::cout << "new_value : " << new_value  << " " << threshold << std::endl;
					img_modified.setvalue(i,j,k,new_value);
				}
				else
					img_modified.setvalue(i,j,k,img_orig.getvalue(i,j,k));
			}

}

//#############################################################################################

void ImageProc::median_filter(Image &img_orig, Image &img_modified, int window_size)
{
	int ws = window_size;
	int kernel_radius = ws / 2;
	int values[ws*ws];
	int index = 0;
	int median_value;

	for(int i = 0; i < img_orig.rows; i++)
	 	for(int j = 0; j < img_orig.cols; j++)
	 		for(int k = 0; k < img_orig.channels; k++)
		 	{
				for(index = 0; index < (ws*ws); index++)
					values[index] = 0;

				index = 0;

				for(int m = -kernel_radius; m <= kernel_radius; m++)
					for(int n = -kernel_radius; n <= kernel_radius; n++)
					{
						if((i+m >= 0) && (i+m < img_orig.rows) && (j+n >= 0) && (j+n < img_orig.cols))
							values[index] = img_orig.getvalue((i+m),(j+n),k);
						else
							values[index] = img_orig.getvalue((i+m),(j-n),k);
						index++;
					}

				mergesort(values, 0, ((ws*ws)-1));
				median_value = get_median(values, (ws*ws));

				img_modified.setvalue(i,j,k,median_value);
			}

}

//#############################################################################################

float ImageProc::get_PSNR(Image &img_orig, Image &img_modified)
{
	float MSE = get_MSE(img_orig, img_modified);
	float Max = 255.0;
	float PSNR = 10.0*log10(pow(Max,2.0)/MSE);
	return PSNR;
}

//#############################################################################################

float ImageProc::get_MSE(Image &img_orig, Image &img_modified)
{
	float sum = 0.0, MSE;
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			sum += pow((img_modified.getvalue(i,j,0) - img_orig.getvalue(i,j,0)),2);

	MSE = sum / (img_modified.rows * img_modified.cols);
	std::cout << MSE << std::endl;
	return MSE;
}

//#############################################################################################

void ImageProc::mergesort(int *a, int low, int high)
{
	int mid;
	if (low < high)
	{
		mid = (low + high) / 2;
		mergesort(a, low, mid);
		mergesort(a, mid + 1, high);
		merge(a, low, high, mid);
	}
	return;
}

//#############################################################################################

void ImageProc::merge(int *a, int low, int high, int mid)
{
	int i, j, k, c[50];
	i = low;
	k = low;
	j = mid + 1;
	while (i <= mid && j <= high)
	{
		if (a[i] < a[j])
		{
			c[k] = a[i];
			k++;	i++;
		}
		else
		{
			c[k] = a[j];
			k++;	j++;
		}
	}
	while (i <= mid)
	{
		c[k] = a[i];
		k++;	i++;
	}
	while (j <= high)
	{
		c[k] = a[j];
		k++;	j++;
	}
	for (i = low; i < k; i++)
	{
		a[i] = c[i];
	}
}

//#############################################################################################

int ImageProc::get_median(int *sorted_array, int size)
{
	// Middle or average of middle values in the sorted array.
	int median = 0.0;
	if ((size % 2) == 0) {
		median = (int)((sorted_array[size / 2] + sorted_array[(size / 2) - 1]) / 2.0);
	}
	else {
		median = sorted_array[size / 2];
	}

	return median;
}

//#############################################################################################
//#############################################################################################


//#############################################################################################
//######### Bilateral Filter ##################################################################

void ImageProc::apply_bilateral_filter(const char *infile, const char *outfile, int width, int height, int window_size, int sigma, int sigma_simi, int iter)
{
	Image img_orig("color", width, height);
	Image img_modified("color", width, height);
	img_orig.read_image(infile, img_orig.cols, img_orig.rows);

	// Image img_red("bw", img_orig.cols, img_orig.rows);
	// Image img_blue("bw", img_orig.cols, img_orig.rows);
	// Image img_green("bw", img_orig.cols, img_orig.rows);
	// Image img_redOut("bw", img_orig.cols, img_orig.rows);
	// Image img_greenOut("bw", img_orig.cols, img_orig.rows);
	// Image img_blueOut("bw", img_orig.cols, img_orig.rows);

	// for(int i = 0; i < img_orig.rows; i++)
	// 	for(int j = 0; j < img_orig.cols; j++)
	// 	{
	// 		img_red.setvalue(i,j,0,img_orig.getvalue(i,j,0));
	// 		img_green.setvalue(i,j,0,img_orig.getvalue(i,j,1));
	// 		img_blue.setvalue(i,j,0,img_orig.getvalue(i,j,2));
	// 	}

	for(int i = 0; i < iter; i++)
	{
		// median_filter(img_orig, img_modified, window_size);
		// for(int i = 0; i < img_orig.rows; i++)
		// for(int j = 0; j < img_orig.cols; j++)
		// for(int k = 0; k < img_orig.channels; k++)
		// 	img_orig.setvalue(i,j,k,img_modified.getvalue(i,j,k));

		bilateral_filter(img_orig, img_modified, window_size, sigma, sigma_simi);
		for(int i = 0; i < img_orig.rows; i++)
		 	for(int j = 0; j < img_orig.cols; j++)
		 		for(int k = 0; k < img_orig.channels; k++)
			 	{
			 		img_orig.setvalue(i,j,k,img_modified.getvalue(i,j,k));
			 	}
	}

	img_modified.write_image(outfile, img_modified.cols, img_modified.rows);
}

//#############################################################################################

void ImageProc::bilateral_filter(Image &img_orig, Image &img_modified, int window_size, int sigma_dist, int sigma_simi)
{
	int ws = window_size;
	double kernel[ws][ws], intensity_diff[ws][ws]; //discrete convolution operator
	double normalization_factor_sum = 0.0;
    int kernel_radius = ws / 2;
    double frac = 0.0;

    for (int row = -kernel_radius; row <= kernel_radius; row++)
        for (int col = -kernel_radius; col <= kernel_radius; col++)
        {
            frac = exp(-1.0 * ((row * row) + (col * col)) / (2.0 * (sigma_dist * sigma_dist)));
            kernel[row + kernel_radius][col + kernel_radius] = frac;
        }


	for(int i = kernel_radius; i < img_orig.rows - kernel_radius; i++)
	 	for(int j = kernel_radius; j < img_orig.cols - kernel_radius; j++)
	 		for(int k = 0; k < img_orig.channels; k++)
		 	{
				float values[ws][ws];
				for(int row = 0; row < ws; row++)
					for(int col = 0; col < ws; col++)
					{
						values[row][col] = 0;
						intensity_diff[row][col] = 0.0;
					}
				int value = 0;
				normalization_factor_sum = 0.0;

				for(int m = -kernel_radius, row = 0; m <= kernel_radius; m++, row++)
					for(int n = -kernel_radius, col = 0; n <= kernel_radius; n++, col++)
					{
						if((i+m >= 0) && (i+m < img_orig.rows) && (j+n >= 0) && (j+n < img_orig.cols))
						{
							intensity_diff[row][col] = exp(-1.0 * ((pow((img_orig.getvalue((i+m),(j+n),k) - img_orig.getvalue(i,j,k)),2.0))
													   /(2 * (sigma_simi*sigma_simi))));
						}
						else
						{
							intensity_diff[row][col] = exp(-1.0 * ((pow((img_orig.getvalue((i-m),(j-n),k) - img_orig.getvalue(i,j,k)),2.0))
													   /(2 * (sigma_simi*sigma_simi))));
						}
//std::cout << intensity_diff[row][col] << " " << img_orig.getvalue(i+m,j+n,k) << " " << img_orig(i,j,k) << " " << i+m << " " << j+n << " " << k << " " << std::endl;
					}

				for(int row = 0; row < ws; row++)
				 	for(int col = 0; col < ws; col++)
				 		normalization_factor_sum += kernel[row][col] * intensity_diff[row][col];
//std::cout << "\nstart" << std::endl;
				for(int m = -kernel_radius, row = 0; m <= kernel_radius; m++, row++)
					for(int n = -kernel_radius, col = 0; n <= kernel_radius; n++, col++)
					{
						if((i+m >= 0) && (i+m < img_orig.rows) && (j+n >= 0) && (j+n < img_orig.cols))
							values[row][col] = img_orig.getvalue((i+m),(j+n),k) * kernel[row][col] * intensity_diff[row][col] / normalization_factor_sum;
						else
							values[row][col] = img_orig.getvalue((i+m),(j-n),k) * kernel[row][col] * intensity_diff[row][col] / normalization_factor_sum;
						//std::cout << values[row][col] << " " << img_orig((i+m),(j+n),k) << " " << kernel[row][col] << " " << intensity_diff[row][col] << " " << normalization_factor_sum << std::endl;
					}

				for(int row = 0; row < ws; row++)
					for(int col = 0; col < ws; col++)
					{
						//std::cout << values[row][col] << std::endl;
						value += values[row][col];
					}

					//std::cout << value << std::endl;
					//std::cout << img_orig.getvalue(i,j,k) << "\n " << std::endl;

				img_modified.setvalue(i,j,k,value);
			}
}

//#############################################################################################
//#############################################################################################


//#############################################################################################
//######### Non Local Mean ####################################################################

void ImageProc::apply_non_local_mean(const char *infile, const char *outfile, int width, int height, int window_size, int region_size, int sigma, int h)
{
	Image img_orig("color", width, height);
	Image img_modified("color", width, height);
	img_orig.read_image(infile, img_orig.cols, img_orig.rows);

	non_local_mean(img_orig, img_modified, window_size, region_size, sigma, h);

	img_modified.write_image(outfile, img_modified.cols, img_modified.rows);
}

//#############################################################################################

void ImageProc::non_local_mean(Image &img_orig, Image &img_modified, int window_size, int region_size, int sigma, int h)
{
	int ws = window_size;
	int wleft = ((-1*ws) + 1) / 2;
	int wright = (( 1*ws) - 1) / 2;
	int rs = region_size;
	double kernel[rs][rs]; //discrete convolution operator
	double kernel_sum = 0.0;
    int kernel_radius = rs / 2;
    //double distance_value[rs*rs];
    double distortion_value[ws*ws], intensity_values[ws*ws], distortion_value_sum;
    double coeff = 1.0 / (sqrt(2.0 * M_PI) * (sigma));
    double frac = 0.0;
    int index = 0;
    //float distance = 0.0;
    double value_sum;

    for (int row = -kernel_radius; row <= kernel_radius; row++)
        for (int col = -kernel_radius; col <= kernel_radius; col++)
        {
            frac = exp(-1.0 * ((row * row) + (col * col)) / (2.0 * (sigma * sigma)));
            kernel[row + kernel_radius][col + kernel_radius] = coeff * frac;
            kernel_sum += kernel[row + kernel_radius][col + kernel_radius];
        }

    for (int i = 0; i < ws; i++)
        for (int j = 0; j < ws; j++)
            kernel[i][j] = kernel[i][j] / kernel_sum; //Normalized



    int M, N, K;
    double distance_sum = 0.0;
    for(int i = 0; i < img_orig.rows; i++)
 		for(int j = 0; j < img_orig.cols; j++)
 			for(int k = 0; k < img_orig.channels; k++)
	 		{
	 			value_sum = 0.0;
	 			distortion_value_sum = 0.0;
	 			for(int iter = 0; iter < ws*ws; iter++)
	 			{
	 				distortion_value[iter] = 0.0;
	 				intensity_values[iter] = 0.0;
	 			}
	 			index = 0;

	 			//std::cout << "\nstart" << std::endl;
	 			for(int m = wleft; m <= wright; m++)
					for(int n = wleft; n <= wright; n++)
					{
	 					distance_sum = 0.0;

						M = i+m; N = j+n; K = k;
						if((M >= 0) && (M < img_orig.rows) && (N >= 0) && (N < img_orig.cols))
						{
							for(int p = -kernel_radius, row = 0; p <= kernel_radius; p++, row++)
								for(int q = -kernel_radius, col = 0; q <= kernel_radius; q++, col++)
								{
									if((i+p >= 0) && (i+p < img_orig.rows) && (j+q >= 0) && (j+q < img_orig.cols) &&
										(M+p >= 0) && (M+p < img_orig.rows) && (N+q >= 0) && (N+q < img_orig.cols))
									{
										//distance_value[index] = kernel[row][col] * ((img_orig.getvalue((i+p),(j+q),k)) - img_orig.getvalue((M+p),(N+q),k))
										distance_sum += kernel[row][col] * (pow((img_orig.getvalue((i+p),(j+q),k))-img_orig.getvalue((M+p),(N+q),K),2));
									}
									else
									{
										//distance_value[index] = kernel[row][col] * ((img_orig.getvalue((i+p),(j-q),k)) - img_orig.getvalue((M+p),(N-q),k))
										distance_sum += kernel[row][col] * (pow((img_orig.getvalue((i+p),(j-q),k))-img_orig.getvalue((M+p),(N-q),K),2));
									}

								}
							intensity_values[index] = img_orig.getvalue(M,N,K);
							distortion_value[index] = exp(-1.0*(distance_sum/pow(h,2.0)));
							//std::cout << distortion_value[index] << std::endl;

							index++;
						}
						else
						{
							index++;
						}

					}

				for(int index = 0; index < ws*ws; index++)
				{
					distortion_value_sum += distortion_value[index];
					//std::cout << distortion_value_sum << " " << distortion_value[i] << " " << intensity_values[i] << std::endl;
				}

				//for(int i = 0; i < ws*ws; i++)

				for(int t = 0; t < ws*ws; t++)
				{
					value_sum += (intensity_values[t]*(distortion_value[t]/distortion_value_sum));
					//std::cout << intensity_values[i] << " " << (distortion_value[i]/distortion_value_sum) << " " << value_sum << std::endl;

				}
					//std::cout << "     " << value_sum << std::endl;


				img_modified.setvalue(i,j,k,value_sum);

			}
}

//#############################################################################################
//#############################################################################################



void ImageProc::hist_equal_proc(Image &img_orig, Image &img_modified)
{
	//Image img_bw("bw", width, height);

	//img_bw.read_image(infile, width, height);

	int histogram_bins[256] = {0};

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			if((img_orig.getvalue(i,j,0) >= 0) && (img_orig.getvalue(i,j,0) < 256))
				histogram_bins[img_orig.getvalue(i,j,0)]++;


	int hist_cumulative[256] = {0};
	hist_cumulative[0] = histogram_bins[0];
	for(int i = 1; i < 256; i++)
		hist_cumulative[i] = hist_cumulative[i-1] + histogram_bins[i];


	for(int i = 0; i < img_modified.rows; i++)
		for(int j = 0; j < img_modified.cols; j++)
			if      (img_orig.getvalue(i,j,0) < 0  ) img_modified.setvalue(i,j,0,0);
			else if (img_orig.getvalue(i,j,0) > 255) img_modified.setvalue(i,j,0,255);
			else    img_modified.setvalue(i,j,0,(255 * hist_cumulative[img_orig.getvalue(i,j,0)] \
															      / hist_cumulative[255]));


	//img_modified.write_image(outfile, width, height);
}

//#############################################################################################
//#############################################################################################


//#############################################################################################
//######### SOBEL EDGEPOINT DETECTOR ##########################################################

void ImageProc::apply_sobel_operator(const char *infile, const char *outfile, int width, int height, int percentage)
{
	Image img_orig("bw", width, height);
	Image img_edge_map("bw", width, height);


	img_orig.read_image(infile, img_orig.cols, img_orig.rows);

	sobel_edge_detector(img_orig, img_edge_map, percentage);

	img_edge_map.write_image(outfile, img_edge_map.cols, img_edge_map.rows);
}

//#############################################################################################

void ImageProc::sobel_edge_detector(Image &img_orig, Image &img_edge_map, int percentage)
{
	Image img_Gx("bw", img_orig.cols, img_orig.rows);
	Image img_Gy("bw", img_orig.cols, img_orig.rows);
	Image img_modified("bw", img_orig.cols, img_orig.rows);
	int **sobel_opX, **sobel_opY;
	sobel_opX = new int *[3];
	sobel_opY = new int *[3];
	for(int i = 0; i < 3; i++)
	{
		sobel_opX[i] = new int[3];
		sobel_opY[i] = new int[3];
	}

	sobel_opX[0][0] = -1;	sobel_opX[0][1] = 0;	sobel_opX[0][2] = 1;
	sobel_opX[1][0] = -2;	sobel_opX[1][1] = 0;	sobel_opX[1][2] = 2;
	sobel_opX[2][0] = -1;	sobel_opX[2][1] = 0;	sobel_opX[2][2] = 1;

	sobel_opY[0][0] = -1;	sobel_opY[0][1] = -2;	sobel_opY[0][2] = -1;
	sobel_opY[1][0] = 0;	sobel_opY[1][1] = 0;	sobel_opY[1][2] = 0;
	sobel_opY[2][0] = 1;	sobel_opY[2][1] = 2;	sobel_opY[2][2] = 1;
	//int sum = 0;
	int Gx[img_orig.rows][img_orig.cols];
	int Gy[img_orig.rows][img_orig.cols];
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			Gx[i][j] = 0;
			Gy[i][j] = 0;
		}
	int minGx = std::numeric_limits<int>::max();
	int minGy = std::numeric_limits<int>::max();
	int maxGx = std::numeric_limits<int>::min();
	int maxGy = std::numeric_limits<int>::min();
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
			{
				if((i-1)>=0 && (i+1)<img_orig.rows && (j-1)>=0 && (j+1)<img_orig.cols)
				{
					Gx[i][j] = func_gradient(sobel_opX, img_orig, i, j, k, 3);
					if(Gx[i][j] < minGx) minGx = Gx[i][j];
					if(Gx[i][j] > maxGx) maxGx = Gx[i][j];

					Gy[i][j] = func_gradient(sobel_opY, img_orig, i, j, k, 3);
					if(Gy[i][j] < minGy) minGy = Gy[i][j];
					if(Gy[i][j] > maxGy) maxGy = Gy[i][j];

				}

			}


	//Normalize Gradient values
	int abs_minGx = abs(minGx);
	int abs_minGy = abs(minGy);
	maxGx = maxGx + abs_minGx;
	maxGy = maxGy + abs_minGy;


	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			Gx[i][j] = Gx[i][j] + abs_minGx;
			Gx[i][j] = (int)(1.0*Gx[i][j]*255/maxGx);
			Gy[i][j] = Gy[i][j] + abs_minGy;
			Gy[i][j] = (int)(1.0*Gy[i][j]*255/maxGy);
		}

	int nGradient[img_orig.rows][img_orig.cols];
	int min_nGradient = std::numeric_limits<int>::max();
	int max_nGradient = std::numeric_limits<int>::min();
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
			{
				if((i-1)>=0 && (i+1)<img_orig.rows && (j-1)>=0 && (j+1)<img_orig.cols)
				{
					nGradient[i][j] = (int)sqrt(Gx[i][j]*Gx[i][j] + Gy[i][j]*Gy[i][j]);
					if(nGradient[i][j] < min_nGradient) min_nGradient = nGradient[i][j];
					if(nGradient[i][j] > max_nGradient) max_nGradient = nGradient[i][j];

					img_Gx.setvalue(i,j,k,Gx[i][j]);
					img_Gy.setvalue(i,j,k,Gy[i][j]);
				}

			}

	int abs_min_nGradient = abs(min_nGradient);
	max_nGradient = max_nGradient + abs_min_nGradient;

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			nGradient[i][j] = nGradient[i][j] + abs_min_nGradient;
			nGradient[i][j] = (int)(1.0*nGradient[i][j]*255/max_nGradient);
			img_modified.setvalue(i,j,0,nGradient[i][j]);
		}

	img_Gx.write_image("../output_images/Sobel_gradientInX.raw", img_Gx.cols, img_Gx.rows);
	img_Gy.write_image("../output_images/Sobel_gradientInY.raw", img_Gy.cols, img_Gy.rows);


	int histogram_bins[256], hist_cumulative[256];
	get_histogram(img_modified, histogram_bins);
	get_cumulative_histogram(histogram_bins, hist_cumulative);

	int threshold = (int)(((100-percentage)/100.0)*hist_cumulative[255]);

	for(int i = 0; i < 256; i++)
	{
		if(hist_cumulative[i] > threshold)
		{
			threshold = i-1;
			break;
		}
	}

	for(int i = 0; i < img_modified.rows; i++)
		for(int j = 0; j < img_modified.cols; j++)
			for(int k = 0; k < img_modified.channels; k++)
			{
				if((i-1)>=0 && (i+1)<img_modified.rows && (j-1)>=0 && (j+1)<img_modified.cols)
				{
					if(img_modified.getvalue(i,j,k) < threshold)
						img_edge_map.setvalue(i,j,k,255);
					else
						img_edge_map.setvalue(i,j,k,0);
				}
			}

	img_modified.write_image("../output_images/Sobel_gradient_map.raw", img_modified.cols, img_modified.rows);
}

//#############################################################################################

int ImageProc::func_gradient(int **operator_mask, Image &img_orig, int x, int y, int z, int window_size)
{
	int result = 0;
	for(int m = -window_size/2, p = 0; m <= window_size/2; m++, p++)
		for(int n = -window_size/2, q = 0; n <= window_size/2; n++, q++)
		{
			result += img_orig.getvalue((x+m),(y+n),z) * operator_mask[p][q];
		}

	return result;
}

//#############################################################################################

void ImageProc::get_histogram(Image &img_orig, int histogram_bins[256])
{
	for(int i = 0; i < 256; i++)
	{
		histogram_bins[i] = 0;
	}

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			if((img_orig.getvalue(i,j,0) >= 0) && (img_orig.getvalue(i,j,0) < 256))
				histogram_bins[img_orig.getvalue(i,j,0)]++;

}

//#############################################################################################

void ImageProc::get_cumulative_histogram(int histogram_bins[256], int hist_cumulative[256])
{
	hist_cumulative[0] = histogram_bins[0];
	for(int i = 1; i < 256; i++)
		hist_cumulative[i] = hist_cumulative[i-1] + histogram_bins[i];
}

//#############################################################################################
//############## LOG OPERATOR #################################################################

void ImageProc::apply_LoG_operator(const char *infile, const char *outfile, int width, int height)
{
	Image img_orig("bw", width, height);
	Image img_modified("bw", width, height);
	img_orig.read_image(infile, img_orig.cols, img_orig.rows);

	LoG_edge_detector(img_orig, img_modified);

	img_modified.write_image(outfile, img_modified.cols, img_modified.rows);

}

//#############################################################################################

void ImageProc::LoG_edge_detector(Image &img_orig, Image &img_modified)
{
	Image img_GradiantLoG("bw", img_orig.cols, img_orig.rows);
	Image img_ternary_map_LoG("bw", img_orig.cols, img_orig.rows);
	int size = 17;
	float **LoG_kernel;
	LoG_kernel = new float *[size];
	for(int i = 0; i < size; i++)
		LoG_kernel[i] = new float[size];

	float sigma = 2;
	int kernel_radius = size/2;
	float coeff = (-1.0)/(M_PI * pow(sigma,4));
	for(int p = -1*kernel_radius, row = 0 ; p <= kernel_radius ; p++, row++)
		for(int q = -1*kernel_radius, col = 0 ; q <= kernel_radius ; q++, col++)
		{
			float value = (-1.0) * (((p*p) + (q*q))/(2*sigma*sigma));
			LoG_kernel[row][col] = coeff * (1+value) * (exp(value));
		}

	float Gradient_LoG[img_orig.rows][img_orig.cols];
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			Gradient_LoG[i][j] = 0.0;
		}
	float minGradient = std::numeric_limits<float>::max();
	float maxGradient = std::numeric_limits<float>::min();
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
			{
				if((i-1)>=0 && (i+1)<img_orig.rows && (j-1)>=0 && (j+1)<img_orig.cols)
				{
					Gradient_LoG[i][j] = func_gradient_float(LoG_kernel, img_orig, i, j, k, size);
					if(Gradient_LoG[i][j] < minGradient) minGradient = Gradient_LoG[i][j];
					if(Gradient_LoG[i][j] > maxGradient) maxGradient = Gradient_LoG[i][j];
				}

			}


	//Normalize Gradient values
	int abs_minGradient = abs(minGradient);
	maxGradient = maxGradient + abs_minGradient;

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			Gradient_LoG[i][j] = Gradient_LoG[i][j] + abs_minGradient;
			Gradient_LoG[i][j] = (int)(1.0*Gradient_LoG[i][j]*255/maxGradient);
		}


	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
				if((i-1)>=0 && (i+1)<img_orig.rows && (j-1)>=0 && (j+1)<img_orig.cols)
				{
					img_modified.setvalue(i,j,k,Gradient_LoG[i][j]);
					img_GradiantLoG.setvalue(i,j,k,Gradient_LoG[i][j]);
				}

	img_GradiantLoG.write_image("../output_images/Gradient_LoG.raw", img_GradiantLoG.cols, img_GradiantLoG.rows);

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
			{
				if((i-1)>=0 && (i+1)<img_orig.rows && (j-1)>=0 && (j+1)<img_orig.cols)
				{
					if(img_modified.getvalue(i,j,k) < 135)
					{
						img_modified.setvalue(i,j,k,64);
						img_ternary_map_LoG.setvalue(i,j,k,64);
					}
					else if(img_modified.getvalue(i,j,k) > 165)
					{
						img_modified.setvalue(i,j,k,192);
						img_ternary_map_LoG.setvalue(i,j,k,192);
					}
					else
					{
						img_modified.setvalue(i,j,k,128);
						img_ternary_map_LoG.setvalue(i,j,k,128);
					}
				}

			}

	img_ternary_map_LoG.write_image("../output_images/ternary_map_LoG.raw", img_ternary_map_LoG.cols, img_ternary_map_LoG.rows);

	Image img_modified_copy("bw", img_modified.cols, img_modified.rows);

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
				img_modified_copy.setvalue(i,j,k,img_modified.getvalue(i,j,k));

	int zero_crossing;
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
			{
				if((i-1)>=0 && (i+1)<img_orig.rows && (j-1)>=0 && (j+1)<img_orig.cols)
				{
					zero_crossing = check_zero_crossing(img_modified_copy, i, j, k);
					if(zero_crossing == 0) 	img_modified.setvalue(i,j,k,255);
					else					img_modified.setvalue(i,j,k,0);
				}

			}

}

//#############################################################################################

float ImageProc::func_gradient_float(float **operator_mask, Image &img_orig, int x, int y, int z, int window_size)
{
	float result = 0.0;
	for(int m = -window_size/2, p = 0; m <= window_size/2; m++, p++)
		for(int n = -window_size/2, q = 0; n <= window_size/2; n++, q++)
		{
			result += (1.0 * img_orig.getvalue((x+m),(y+n),z) * operator_mask[p][q]);
		}

	return result;
}
//#############################################################################################

int ImageProc::check_zero_crossing(Image &img_modified_copy, int i, int j, int k)
{
	if(img_modified_copy.getvalue(i,j,k) != 128)
	{
	 	return (0);
	}
	else if(	(img_modified_copy.getvalue((i),(j-1),k) == 64 && img_modified_copy.getvalue((i),(j+1),k) == 192)
			||	(img_modified_copy.getvalue((i-1),(j-1),k) == 64 && img_modified_copy.getvalue((i+1),(j+1),k) == 192)
			||	(img_modified_copy.getvalue((i-1),(j),k) == 64 && img_modified_copy.getvalue((i+1),(j),k) == 192)
			||	(img_modified_copy.getvalue((i-1),(j+1),k) == 64 && img_modified_copy.getvalue((i+1),(j-1),k) == 192)
			||	(img_modified_copy.getvalue((i),(j-1),k) == 192 && img_modified_copy.getvalue((i),(j+1),k) == 64)
			||	(img_modified_copy.getvalue((i-1),(j-1),k) == 192 && img_modified_copy.getvalue((i+1),(j+1),k) == 64)
			||	(img_modified_copy.getvalue((i-1),(j),k) == 192 && img_modified_copy.getvalue((i+1),(j),k) == 64)
			||	(img_modified_copy.getvalue((i-1),(j+1),k) == 192 && img_modified_copy.getvalue((i+1),(j-1),k) == 64))
		return (1);
	else return(0);
}

//#############################################################################################
//############# HALFTONING ####################################################################

void ImageProc::halftoning(const char *infile, int width, int height)
{
	Image img_orig("bw", width, height);
	Image img_halftone_fixed_thresh("bw", width, height);
	Image img_halftone_random_thresh("bw", width, height);
	Image img_halftone_dither_matrix_2("bw", width, height);
	Image img_halftone_dither_matrix_4("bw", width, height);
	Image img_halftone_error_diffusion("bw", width, height);
	img_orig.read_image(infile, img_orig.cols, img_orig.rows);

	halftone_fixed_thresh(img_orig, img_halftone_fixed_thresh);
	halftone_random_thresh(img_orig, img_halftone_random_thresh);
	halftone_dither_matrix(img_orig, img_halftone_dither_matrix_2, 2);
	halftone_dither_matrix(img_orig, img_halftone_dither_matrix_4, 4);
	error_diffusion(img_orig, img_halftone_error_diffusion);

	img_halftone_fixed_thresh.write_image("../output_images/halftone_fixed_thresh.raw", img_halftone_fixed_thresh.cols, img_halftone_fixed_thresh.rows);
	img_halftone_random_thresh.write_image("../output_images/halftone_random_thresh.raw", img_halftone_random_thresh.cols, img_halftone_random_thresh.rows);
	img_halftone_dither_matrix_2.write_image("../output_images/halftone_dither_matrix_2x2.raw", img_halftone_dither_matrix_2.cols, img_halftone_dither_matrix_2.rows);
	img_halftone_dither_matrix_4.write_image("../output_images/halftone_dither_matrix_4x4.raw", img_halftone_dither_matrix_4.cols, img_halftone_dither_matrix_4.rows);
	img_halftone_error_diffusion.write_image("../output_images/halftone_error_diffusion.raw", img_halftone_error_diffusion.cols, img_halftone_error_diffusion.rows);
	//img_modified.write_image(outfile, img_modified.cols, img_modified.rows);
}

//#############################################################################################

void ImageProc::halftone_fixed_thresh(Image &img_orig, Image &img_modified)
{
	for(int i = 0; i < img_modified.rows; i++)
		for(int j = 0; j < img_modified.cols; j++)
			for(int k = 0; k < img_modified.channels; k++)
				if(img_orig.getvalue(i,j,k) <= 127)	img_modified.setvalue(i,j,k,0  );
				else								img_modified.setvalue(i,j,k,255);
}

//#############################################################################################

void ImageProc::halftone_random_thresh(Image &img_orig, Image &img_modified)
{
	srand(time(0));
	for(int i = 0; i < img_modified.rows; i++)
		for(int j = 0; j < img_modified.cols; j++)
			for(int k = 0; k < img_modified.channels; k++)
				if(img_orig.getvalue(i,j,k) <= rand() % 255)	img_modified.setvalue(i,j,k,0  );
				else											img_modified.setvalue(i,j,k,255);
}

//#############################################################################################

void ImageProc::halftone_dither_matrix(Image &img_orig, Image &img_modified, const int matrix_size)
{
	int dither_matrix[matrix_size][matrix_size];
	float T[matrix_size][matrix_size];
	if(matrix_size == 2)
	{
		dither_matrix[0][0] = 1;	dither_matrix[0][1] = 2;
		dither_matrix[1][0] = 3;	dither_matrix[1][1] = 0;
	}
	if(matrix_size == 4)
	{
		dither_matrix[0][0] = 5;	dither_matrix[0][1] = 9;	dither_matrix[0][2] = 6;	dither_matrix[0][3] = 10;
		dither_matrix[1][0] = 13;	dither_matrix[1][1] = 1;	dither_matrix[1][2] = 14;	dither_matrix[1][3] = 2;
		dither_matrix[2][0] = 7;	dither_matrix[2][1] = 11;	dither_matrix[2][2] = 4;	dither_matrix[2][3] = 8;
		dither_matrix[3][0] = 15;	dither_matrix[3][1] = 3;	dither_matrix[3][2] = 12;	dither_matrix[3][3] = 0;
	}

	for(int i = 0; i < matrix_size; i++)
		for(int j = 0; j < matrix_size; j++)
			T[i][j] = ((1.0 * dither_matrix[i][j]) + 0.5) / (matrix_size*matrix_size);


	for(int i = 0; i < img_modified.rows; i++)
		for(int j = 0; j < img_modified.cols; j++)
			for(int k = 0; k < img_modified.channels; k++)
				if(float(img_orig.getvalue(i,j,k)/255.0) <= T[i%matrix_size][j%matrix_size])
					img_modified.setvalue(i,j,k,0  );
				else
					img_modified.setvalue(i,j,k,255);

}

//#############################################################################################

void ImageProc::error_diffusion(Image &img_orig, Image &img_modified)
{
	float diffusionError;
	int direction;
	float img_orig_float[img_orig.rows][img_orig.cols][img_orig.channels];

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
			{
				img_orig_float[i][j][k] = float(img_orig.getvalue(i,j,k)/255.0);
			}


	for(int i = 0; i < img_modified.rows-1; i++)
	{
		if(i%2 == 0)
		{
			direction = LEFT2RIGHT;
			for(int j = 0; j < img_modified.cols; j++)
				for(int k = 0; k < img_modified.channels; k++)
				{

					if(img_orig_float[i][j][k] <= 0.5)
					{
						diffusionError = img_orig_float[i][j][k] - 0.0;
						img_modified.setvalue(i,j,k,0);
					}
					else
					{
						diffusionError = img_orig_float[i][j][k] - 1.0;
						img_modified.setvalue(i,j,k,255);
					}

					img_orig_float[i][j+1][k] = img_orig_float[i][j+1][k]+(diffusionError*7.0/16);
					img_orig_float[i+1][j+1][k] = img_orig_float[i+1][j+1][k]+(diffusionError*1.0/16);
					img_orig_float[i+1][j][k] = img_orig_float[i+1][j][k]+(diffusionError*5.0/16);
					img_orig_float[i+1][j-1][k] = img_orig_float[i+1][j-1][k]+(diffusionError*3.0/16);

				}
		}

		else
		{
			direction = RIGHT2LEFT;
			for(int j = img_modified.cols - 1; j >= 0; j--)
				for(int k = 0; k < img_modified.channels; k++)
				{
					if(img_orig_float[i][j][k] <= 0.5)
					{
						diffusionError = img_orig_float[i][j][k] - 0.0;
						img_modified.setvalue(i,j,k,0);
					}
					else
					{
						diffusionError = img_orig_float[i][j][k] - 1.0;
						img_modified.setvalue(i,j,k,255);
					}

					img_orig_float[i][j-1][k] = img_orig_float[i][j-1][k]+(diffusionError*7.0/16);
					img_orig_float[i+1][j-1][k] = img_orig_float[i+1][j-1][k]+(diffusionError*1.0/16);
					img_orig_float[i+1][j][k] = img_orig_float[i+1][j][k]+(diffusionError*5.0/16);
					img_orig_float[i+1][j+1][k] = img_orig_float[i+1][j+1][k]+(diffusionError*3.0/16);
				}

		}


	}

}

//#############################################################################################

void ImageProc::color_halftoning(const char *infile, const char *outfile, int width, int height)
{
	Image img_orig("color", width, height);
	Image img_modified("color", width, height);
	img_orig.read_image(infile, img_orig.cols, img_orig.rows);

	error_diffusion(img_orig, img_modified);
	//mbvq_error_diffusion(img_orig, img_modified);

	img_modified.write_image(outfile, img_modified.cols, img_modified.rows);
}

//#############################################################################################

void ImageProc::mbvq_error_diffusion(Image &img_orig, Image &img_modified)
{
	float diffusionError[3], distance[6];
	int direction, quad, vertex_number;
	float img_orig_float[img_orig.rows][img_orig.cols][img_orig.channels];

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
			{
				img_orig_float[i][j][k] = float(img_orig.getvalue(i,j,k)/255.0);
			}


	for(int i = 0; i < img_modified.rows-1; i++)
	{
		if(i%2 == 0)
		{
			direction = LEFT2RIGHT;
			for(int j = 0; j < img_modified.cols; j++)
			{
				int R = img_orig.getvalue(i,j,0);
				int G = img_orig.getvalue(i,j,1);
				int B = img_orig.getvalue(i,j,2);
				int r = img_orig_float[i][j][0];
				int g = img_orig_float[i][j][1];
				int b = img_orig_float[i][j][2];
				if((R+G) > 255)
					if((G+B) > 255)
						if((R+G+B) > 510) 	quad = CMYW;
						else 				quad = MYGC;
					else quad = RGMY;
				else if(!((G+B) > 255))
					if(!((R+G+B) > 255)) 	quad = KRGB;
					else 					quad = RGBM;
				else quad = CMBG;

				distance[0] = distance_from_vertex(r,g,b,'C');
				distance[1] = distance_from_vertex(r,g,b,'M');
				distance[2] = distance_from_vertex(r,g,b,'Y');
				distance[3] = distance_from_vertex(r,g,b,'W');
				distance[4] = distance_from_vertex(r,g,b,'R');
				distance[5] = distance_from_vertex(r,g,b,'G');
				distance[6] = distance_from_vertex(r,g,b,'B');
				distance[7] = distance_from_vertex(r,g,b,'K');

				if(quad == CMYW)
					vertex_number = nearest_vertex(distance,0,1,2,3);
				else if(quad == MYGC)
					vertex_number = nearest_vertex(distance,1,2,5,0);
				else if(quad == RGMY)
					vertex_number = nearest_vertex(distance,4,5,1,2);
				else if(quad == CMBG)
					vertex_number = nearest_vertex(distance,0,1,6,5);
				else if(quad == RGBM)
					vertex_number = nearest_vertex(distance,4,5,6,1);
				else if(quad == KRGB)
					vertex_number = nearest_vertex(distance,7,4,5,6);

				switch(vertex_number)
				{
					case 0:
						diffusionError[0] = r - 1.0;
						diffusionError[1] = g - 0.0;
						diffusionError[2] = b - 1.0;
						img_modified.setvalue(i,j,0,255);
						img_modified.setvalue(i,j,1,0);
						img_modified.setvalue(i,j,2,255);
						break;
					case 1:
						diffusionError[0] = r - 0.0;
						diffusionError[1] = g - 1.0;
						diffusionError[2] = b - 1.0;
						img_modified.setvalue(i,j,0,0);
						img_modified.setvalue(i,j,1,255);
						img_modified.setvalue(i,j,2,255);
						break;
					case 2:
						diffusionError[0] = r - 1.0;
						diffusionError[1] = g - 1.0;
						diffusionError[2] = b - 0.0;
						img_modified.setvalue(i,j,0,255);
						img_modified.setvalue(i,j,1,255);
						img_modified.setvalue(i,j,2,0);
						break;
					case 3:
						diffusionError[0] = r - 1.0;
						diffusionError[1] = g - 1.0;
						diffusionError[2] = b - 1.0;
						img_modified.setvalue(i,j,0,255);
						img_modified.setvalue(i,j,1,255);
						img_modified.setvalue(i,j,2,255);
						break;
					case 4:
						diffusionError[0] = r - 0.0;
						diffusionError[1] = g - 1.0;
						diffusionError[2] = b - 0.0;
						img_modified.setvalue(i,j,0,0);
						img_modified.setvalue(i,j,1,255);
						img_modified.setvalue(i,j,2,0);
						break;
					case 5:
						diffusionError[0] = r - 1.0;
						diffusionError[1] = g - 0.0;
						diffusionError[2] = b - 0.0;
						img_modified.setvalue(i,j,0,255);
						img_modified.setvalue(i,j,1,0);
						img_modified.setvalue(i,j,2,0);
						break;
					case 6:
						diffusionError[0] = r - 0.0;
						diffusionError[1] = g - 0.0;
						diffusionError[2] = b - 1.0;
						img_modified.setvalue(i,j,0,0);
						img_modified.setvalue(i,j,1,0);
						img_modified.setvalue(i,j,2,255);
						break;
					case 7:
						diffusionError[0] = r - 0.0;
						diffusionError[1] = g - 0.0;
						diffusionError[2] = b - 0.0;
						img_modified.setvalue(i,j,0,0);
						img_modified.setvalue(i,j,1,0);
						img_modified.setvalue(i,j,2,0);
						break;
					default:
						break;
				}

				for(int k = 0; k < 3; k++)
				{
					img_orig_float[i][j+1][k] = img_orig_float[i][j+1][k]+(diffusionError[k]*7.0/16);
					img_orig_float[i+1][j+1][k] = img_orig_float[i+1][j+1][k]+(diffusionError[k]*1.0/16);
					img_orig_float[i+1][j][k] = img_orig_float[i+1][j][k]+(diffusionError[k]*5.0/16);
					img_orig_float[i+1][j-1][k] = img_orig_float[i+1][j-1][k]+(diffusionError[k]*3.0/16);
				}

			}
		}

		else
		{
			direction = RIGHT2LEFT;
			for(int j = img_modified.cols - 1; j >= 0; j--)
			{
				int R = img_orig.getvalue(i,j,0);
				int G = img_orig.getvalue(i,j,1);
				int B = img_orig.getvalue(i,j,2);
				int r = img_orig_float[i][j][0];
				int g = img_orig_float[i][j][1];
				int b = img_orig_float[i][j][2];
				if((R+G) > 255)
					if((G+B) > 255)
						if((R+G+B) > 510) 	quad = CMYW;
						else 				quad = MYGC;
					else quad = RGMY;
				else if(!((G+B) > 255))
					if(!((R+G+B) > 255)) 	quad = KRGB;
					else 					quad = RGBM;
				else quad = CMBG;

				distance[0] = distance_from_vertex(r,g,b,'C');
				distance[1] = distance_from_vertex(r,g,b,'M');
				distance[2] = distance_from_vertex(r,g,b,'Y');
				distance[3] = distance_from_vertex(r,g,b,'W');
				distance[4] = distance_from_vertex(r,g,b,'R');
				distance[5] = distance_from_vertex(r,g,b,'G');
				distance[6] = distance_from_vertex(r,g,b,'B');
				distance[7] = distance_from_vertex(r,g,b,'K');

				if(quad == CMYW)
					vertex_number = nearest_vertex(distance,0,1,2,3);
				else if(quad == MYGC)
					vertex_number = nearest_vertex(distance,1,2,5,0);
				else if(quad == RGMY)
					vertex_number = nearest_vertex(distance,4,5,1,2);
				else if(quad == CMBG)
					vertex_number = nearest_vertex(distance,0,1,6,5);
				else if(quad == RGBM)
					vertex_number = nearest_vertex(distance,4,5,6,1);
				else if(quad == KRGB)
					vertex_number = nearest_vertex(distance,7,4,5,6);

				switch(vertex_number)
				{
					case 0:
						diffusionError[0] = r - 1.0;
						diffusionError[1] = g - 0.0;
						diffusionError[2] = b - 1.0;
						img_modified.setvalue(i,j,0,255);
						img_modified.setvalue(i,j,1,0);
						img_modified.setvalue(i,j,2,255);
						break;
					case 1:
						diffusionError[0] = r - 0.0;
						diffusionError[1] = g - 1.0;
						diffusionError[2] = b - 1.0;
						img_modified.setvalue(i,j,0,0);
						img_modified.setvalue(i,j,1,255);
						img_modified.setvalue(i,j,2,255);
						break;
					case 2:
						diffusionError[0] = r - 1.0;
						diffusionError[1] = g - 1.0;
						diffusionError[2] = b - 0.0;
						img_modified.setvalue(i,j,0,255);
						img_modified.setvalue(i,j,1,255);
						img_modified.setvalue(i,j,2,0);
						break;
					case 3:
						diffusionError[0] = r - 1.0;
						diffusionError[1] = g - 1.0;
						diffusionError[2] = b - 1.0;
						img_modified.setvalue(i,j,0,255);
						img_modified.setvalue(i,j,1,255);
						img_modified.setvalue(i,j,2,255);
						break;
					case 4:
						diffusionError[0] = r - 0.0;
						diffusionError[1] = g - 1.0;
						diffusionError[2] = b - 0.0;
						img_modified.setvalue(i,j,0,0);
						img_modified.setvalue(i,j,1,255);
						img_modified.setvalue(i,j,2,0);
						break;
					case 5:
						diffusionError[0] = r - 1.0;
						diffusionError[1] = g - 0.0;
						diffusionError[2] = b - 0.0;
						img_modified.setvalue(i,j,0,255);
						img_modified.setvalue(i,j,1,0);
						img_modified.setvalue(i,j,2,0);
						break;
					case 6:
						diffusionError[0] = r - 0.0;
						diffusionError[1] = g - 0.0;
						diffusionError[2] = b - 1.0;
						img_modified.setvalue(i,j,0,0);
						img_modified.setvalue(i,j,1,0);
						img_modified.setvalue(i,j,2,255);
						break;
					case 7:
						diffusionError[0] = r - 0.0;
						diffusionError[1] = g - 0.0;
						diffusionError[2] = b - 0.0;
						img_modified.setvalue(i,j,0,0);
						img_modified.setvalue(i,j,1,0);
						img_modified.setvalue(i,j,2,0);
						break;
					default:
						break;
				}

				for(int k = 0; k < 3; k++)
				{
					img_orig_float[i][j-1][k] = img_orig_float[i][j-1][k]+(diffusionError[k]*7.0/16);
					img_orig_float[i+1][j-1][k] = img_orig_float[i+1][j-1][k]+(diffusionError[k]*1.0/16);
					img_orig_float[i+1][j][k] = img_orig_float[i+1][j][k]+(diffusionError[k]*5.0/16);
					img_orig_float[i+1][j+1][k] = img_orig_float[i+1][j+1][k]+(diffusionError[k]*3.0/16);
				}
			}

		}
	}
}

//#############################################################################################

float ImageProc::distance_from_vertex(float r, float g, float b, char vertex)
{
	float coord[3];
	if(vertex == 'C')
	{
		coord[0] = 1.0; coord[1] = 0.0; coord[2] = 1.0;
	}
	else if(vertex = 'M')
	{
		coord[0] = 0.0; coord[1] = 1.0; coord[2] = 1.0;
	}
	else if(vertex = 'Y')
	{
		coord[0] = 1.0; coord[1] = 1.0; coord[2] = 0.0;
	}
	else if(vertex = 'W')
	{
		coord[0] = 1.0; coord[1] = 1.0; coord[2] = 1.0;
	}
	else if(vertex = 'R')
	{
		coord[0] = 0.0; coord[1] = 1.0; coord[2] = 0.0;
	}
	else if(vertex = 'G')
	{
		coord[0] = 1.0; coord[1] = 0.0; coord[2] = 0.0;
	}
	else if(vertex = 'B')
	{
		coord[0] = 0.0; coord[1] = 0.0; coord[2] = 1.0;
	}
	else if(vertex = 'K')
	{
		coord[0] = 0.0; coord[1] = 0.0; coord[2] = 0.0;
	}

	float distance;
	distance 	= pow((r-coord[0]),2)
				+ pow((g-coord[1]),2)
				+ pow((b-coord[2]),2);
	return distance;
}

//#############################################################################################

int ImageProc::nearest_vertex(float distance[8], int i, int j, int k, int l)
{
	if 		(distance[i] < distance[j] && distance[i] < distance[k] && distance[i] < distance[l])	return i;
	else if (distance[j] < distance[i] && distance[i] < distance[k] && distance[i] < distance[l])	return j;
	else if (distance[k] < distance[i] && distance[i] < distance[j] && distance[i] < distance[l])	return k;
	else if (distance[l] < distance[i] && distance[i] < distance[j] && distance[i] < distance[k])	return l;
}

//#############################################################################################
//#############################################################################################


//#############################################################################################
//############# ARTISTIC EFFECT ###############################################################

void ImageProc::apply_artistic_effect(const char *infile, const char *outfile, int width, int height, float tau, float epsilon, float phi, int window_size, int sigma_dist, int sigma_simi, int iter)
{
	Image img_orig("color", width, height);
	Image img_orig_rgb("color", width, height);
	Image img_orig_bw("bw", width, height);
	Image img_orig_bw_bf("bw", width, height);
	Image img_modified("bw", width, height);
	Image img_bilateral_color("color", width, height);
	Image img_modified_color("color", width, height);
	img_orig.read_image(infile, img_orig.cols, img_orig.rows);

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			img_orig_bw.setvalue(i,j,0,((img_orig.getvalue(i,j,0)*0.21)\
								  	   +(img_orig.getvalue(i,j,1)*0.72)\
								  	   +(img_orig.getvalue(i,j,2)*0.07)));

	//rgb2cielab(img_orig, img_modified);

	//	//Bilateral Filter
	for(int i = 0; i < iter; i++)
	{
		bilateral_filter(img_orig, img_bilateral_color, window_size, sigma_dist, sigma_simi);
		for(int i = 0; i < img_orig.rows; i++)
		 	for(int j = 0; j < img_orig.cols; j++)
		 		for(int k = 0; k < img_orig.channels; k++)
			 	{
			 		img_orig.setvalue(i,j,k,img_bilateral_color.getvalue(i,j,k));
			 	}
	}

	img_bilateral_color.write_image("../output_images/bilateral_art_effect.raw", img_bilateral_color.cols, img_bilateral_color.rows);
	//cielab2rgb(img_modified_color2, img_orig_rgb);

	//img_modified_color.write_image(outfile, img_modified.cols, img_modified.rows);

	for(int i = 0; i < img_orig_bw.rows; i++)
		for(int j = 0; j < img_orig_bw.cols; j++)
			for(int k = 0; k < img_orig_bw.channels; k++)
				img_orig_bw.setvalue(i,j,k,((img_orig.getvalue(i,j,0)*0.21)\
									  +(img_orig.getvalue(i,j,1)*0.72)\
									  +(img_orig.getvalue(i,j,2)*0.07)));

	Image img_sigma_small("bw", width, height);
	Image img_sigma_large("bw", width, height);
	Image img_xDoG("bw", width, height);

	int sigma = 1;

	gaussian_filter(img_orig_bw, img_sigma_small, 9, sigma);
	gaussian_filter(img_orig_bw, img_sigma_large, 9, sigma*sqrt(1.6));

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
				img_xDoG.setvalue(i,j,k,(int)((float)img_sigma_small.getvalue(i,j,k) - (float)(1.0*tau*img_sigma_large.getvalue(i,j,k))));
				//img_xDoG.setvalue(i,j,k,(int)((float)img_sigma_small.getvalue(i,j,k) - (float)(1.0*img_sigma_large.getvalue(i,j,k))));
	img_xDoG.write_image("../output_images/xDoG_art_effect.raw", img_xDoG.cols, img_xDoG.rows);

	float xDoG_val, value;
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
			{
				value = (float)img_xDoG.getvalue(i,j,k)/255.0;
				if(value > epsilon)
					img_modified.setvalue(i,j,k,0);
				else
				{
					value = 1.0 + tanh(phi*(((float)img_sigma_small.getvalue(i,j,k) - (float)(1.0*tau*img_sigma_large.getvalue(i,j,k))) - epsilon));
					img_modified.setvalue(i,j,k,(int)(value*255.0));
				}

			}

	img_modified.write_image("../output_images/contours_art_effect.raw", img_modified.cols, img_modified.rows);

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
				if(img_modified.getvalue(i,j,k) == 0)
					img_modified.setvalue(i,j,k,0);
				else
					img_modified.setvalue(i,j,k,1);

//	for(int i = 0; i < img_orig.rows; i++)
//		for(int j = 0; j < img_orig.cols; j++)
//			for(int k = 0; k < img_orig.channels; k++)
//				if(img_xDoG.getvalue(i,j,k) == 255)
//					img_modified.setvalue(i,j,k,0);
//				else
//					img_modified.setvalue(i,j,k,1);


	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
				img_modified_color.setvalue(i,j,k,(int)((img_bilateral_color.getvalue(i,j,k) * (img_modified.getvalue(i,j,0)))));
	img_modified_color.write_image(outfile, img_modified.cols, img_modified.rows);


}

//#############################################################################################

void ImageProc::rgb2cielab(Image &img_orig, Image &img_modified)
{
	double L,A,B,r,g,b;
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			r = (double)(img_orig.getvalue(i,j,0)/255.0);
			g = (double)(img_orig.getvalue(i,j,1)/255.0);
			b = (double)(img_orig.getvalue(i,j,2)/255.0);
			Rgb2Lab(&L,&A,&B,r,g,b);
			//std::cout << L << " " << A << " " << B <<std::endl;
			img_modified.setvalue(i,j,0,(unsigned char)(L));
			img_modified.setvalue(i,j,1,(unsigned char)(A));
			img_modified.setvalue(i,j,2,(unsigned char)(B));
			//std::cout << img_modified.getvalue(i,j,0) << " " << img_modified.getvalue(i,j,1) << " " << img_modified.getvalue(i,j,2)<<std::endl;
		}
}

//#############################################################################################

void ImageProc::cielab2rgb(Image &img_orig, Image &img_modified)
{
	double L,A,B,r,g,b;
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			L = (double)(img_orig.getvalue(i,j,0));
			A = (double)(img_orig.getvalue(i,j,1));
			B = (double)(img_orig.getvalue(i,j,2));
			Lab2Rgb(&r,&g,&b,L,A,B);
			std::cout << L << " " << A << " " << B <<std::endl;
			img_modified.setvalue(i,j,0,(unsigned char)(r*255));
			img_modified.setvalue(i,j,1,(unsigned char)(g*255));
			img_modified.setvalue(i,j,2,(unsigned char)(b*255));
			std::cout << img_modified.getvalue(i,j,0) << " " << img_modified.getvalue(i,j,1) << " " << img_modified.getvalue(i,j,2)<<std::endl;
		}
}
//#############################################################################################
//#############################################################################################

//#############################################################################################
//################# MORPHOLOGY ################################################################

void ImageProc::apply_morphology(const char *infile, const char *outfile, int width, int height)
{
	Image img_orig_bw("bw", width, height);
	Image img_binary("bw", width, height);
	Image img_shrinked("bw", width, height);
	Image img_thinned("bw", width, height);
	Image img_skeletonized("bw", width, height);
	Image img_unsharped("bw", width, height);
	Image img_modified("bw", width, height);
	Image img_modified_color("color", width, height);

	img_orig_bw.read_image(infile, img_orig_bw.cols, img_orig_bw.rows);

	//unsharp_mask(img_orig_bw, img_unsharp_masked);
	otsu_th(img_orig_bw, img_binary);
	img_binary.write_image("../output_images/morphology_binarized_img.raw", img_binary.cols, img_binary.rows);

	shrink(img_binary, img_shrinked);
	thin(img_binary, img_thinned);
	skeletonize(img_binary, img_skeletonized);

	img_shrinked.write_image("../output_images/morphology_shrinked_img.raw", img_shrinked.cols, img_shrinked.rows);
	img_thinned.write_image("../output_images/morphology_thinned_img.raw", img_thinned.cols, img_thinned.rows);
	img_skeletonized.write_image("../output_images/morphology_skeletonized_img.raw", img_skeletonized.cols, img_skeletonized.rows);
}

//#############################################################################################

void ImageProc::apply_binarization(const char* infile, const char *outfile, int width, int height)
{
	Image img_orig_bw("bw", width, height);
	Image img_binary("bw", width, height);

	img_orig_bw.read_image(infile, img_orig_bw.cols, img_orig_bw.rows);
	//unsharp_mask(img_orig_bw, img_unsharp_masked);
	otsu_th(img_orig_bw, img_binary);
	img_binary.write_image(outfile, img_binary.cols, img_binary.rows);
}

//#############################################################################################

void ImageProc::apply_minutae_extraction(const char* infile, const char *outfile, int width, int height)
{
	Image img_orig_bw("bw", width, height);
	Image img_minutae("bw", width, height);


	img_minutae.write_image(outfile, img_minutae.cols, img_minutae.rows);
}

//#############################################################################################

void ImageProc::minutae_extraction(Image &img_orig, Image &img_modified)
{


}


//#############################################################################################

void ImageProc::otsu_th(Image &img_orig, Image &img_binary)
{
  int levels = 256;
  int hist[levels];
  double prob[levels], omega[levels]; //prob of graylevels
  double myu[levels];   //mean value for separation
  double max_sigma, sigma[levels]; //inter-class variance
  int i, threshold;

  for (i = 0; i < levels; i++) hist[i] = 0;
  for (int i = 0; i < img_orig.rows; i++)
    for (int j = 0; j < img_orig.cols; j++)
    {
      hist[img_orig.getvalue(i,j,0)]++;
    }


  for ( i = 0; i < levels; i ++ ) 
  {
    prob[i] = (double)hist[i] / (img_orig.rows * img_orig.cols);
  }


  omega[0] = prob[0];
  myu[0] = 0.0;
  for (i = 1; i < levels; i++) {
    omega[i] = omega[i-1] + prob[i];
    myu[i] = myu[i-1] + i*prob[i];

  }


  //threshold = 0;
  max_sigma = 0.0;
  for (i = 0; i < levels-1; i++) {
    if (omega[i] != 0.0 && omega[i] != 1.0)
      sigma[i] = pow(myu[levels-1]*omega[i] - myu[i], 2) /
	(omega[i]*(1.0 - omega[i]));
    else
      sigma[i] = 0.0;
    if (sigma[i] > max_sigma) {
      max_sigma = sigma[i];
      threshold = i;
    }
  }

   printf("\nthreshold value = %d\n", threshold);


  for (int i = 0; i < img_binary.rows; i++)
    for (int j = 0; j < img_binary.cols; j++)
      if (img_orig.getvalue(i,j,0) > threshold)
		img_binary.setvalue(i,j,0,255);
      else
		img_binary.setvalue(i,j,0,0);

}

//#############################################################################################

void ImageProc::shrink(Image &img_binary, Image &img_shrinked)
{
	Image img_after_first("bw", img_binary.rows, img_binary.cols);
	Image img_count_ridge_length("bw", img_binary.rows, img_binary.cols);

	for(int i = 0; i < img_binary.rows; i++)
		for(int j = 0; j < img_binary.cols; j++)
			for(int k = 0; k < img_binary.channels; k++)
				img_count_ridge_length.setvalue(i,j,k,255);


	int index  = 0, superflag = 1, flag = 0, black_pixel_count = 0, pixel_count = 0;
	std::string target;
	//char *target_char_arr = new char[9];

	for(int i = 0; i < img_binary.rows; i++)
		for(int j = 0; j < img_binary.cols; j++)
			for(int k = 0; k < img_binary.channels; k++)
				img_shrinked.setvalue(i,j,k,img_binary.getvalue(i,j,k));

	for(int i = 0+1; i < img_shrinked.rows-1; i++)
		for(int j = 0+1; j < img_shrinked.cols-1; j++)
			for(int k = 0; k < img_shrinked.channels; k++)
				if(img_shrinked.getvalue(i,j,k) == 0) pixel_count++;
	black_pixel_count = pixel_count;

	

	while(superflag == 1)
	{
		for(int i = 0; i < img_shrinked.rows; i++)
			for(int j = 0; j < img_shrinked.cols; j++)
				for(int k = 0; k < img_shrinked.channels; k++)
					img_after_first.setvalue(i,j,k,255);

		for(int i = 0+1; i < img_shrinked.rows-1; i++)
			for(int j = 0+1; j < img_shrinked.cols-1; j++)
				for(int k = 0; k < img_shrinked.channels; k++)
				{
					if(img_shrinked.getvalue(i,j,k) == 0)
					{
						index = -1; flag = 0;
						target.clear();
						for(int p = i-1; p <= i+1; p++)
							for(int q = j-1; q <= j+1; q++)
							{
								//std::cout << img_shrinked.getvalue(p,q,k)/255 << std::endl;
								if(!(p==i && q==j))
									if(img_shrinked.getvalue(p,q,0) == 0)
										target.append("1");
									else
										target.append("0");
								else
									continue;
							}
						//target_char_arr[8] = NULL;
						//target(target_char_arr);
						//std::cout << "target 1:" << target << std::endl;

						//std::string target_str(target);
						for(int t = 0; t < SHRINK_FIRST_PATTERNS; t++)
						{
							std::string testpattern(shrink_first[t]);
							flag = 0;
							if(testpattern.find('D') == std::string::npos)
							{
								if(target.compare(testpattern) == 0)
								{
									//std::cout << "pattern match" << std::endl;
									flag = 1;
								}
								else
								{
									//std::cout << "pattern nomatch" << std::endl;
									flag = 0;
								}
							}
							else
							{
								for(int j = 0; j < 8; j++)
								{
									if(testpattern[j] == 'D')
										continue;
									else if(target[j] != testpattern[j])
									{
										//std::cout << target << " hey " << testpattern << std::endl;
										flag = 0;
										break;
									}
									else
										flag = 1;
								}
							}

							if(flag == 1)
								break;
							else
								continue;

						}

						if(flag == 1)
							img_after_first.setvalue(i,j,k,0*255);
						else
							img_after_first.setvalue(i,j,k,1*255);

					}
				}

			for(int i = 0+1; i < img_after_first.rows-1; i++)
				for(int j = 0+1; j < img_after_first.cols-1; j++)
					for(int k = 0; k < img_after_first.channels; k++)
					{
						if(img_after_first.getvalue(i,j,k) == 0)
						{
							target.clear();
							index = -1; flag = 0;
							for(int p = i-1; p <= i+1; p++)
								for(int q = j-1; q <= j+1; q++)
								{
									if(!(p==i && q==j))
										if(img_after_first.getvalue(p,q,0) == 0)
											target.append("M");
										else
											target.append("0");
								}

							//std::cout << "target1: " << target;

							for(int t = 0; t < SHRINK_THIN_SECOND_PATTERNS; t++)
							{
								std::string testpattern(shrink__thin_second[t]);
								flag = 1;
								//std::cout << " " << testpattern << std::endl;
								if(testpattern.find('D') == std::string::npos)
								{

									if(target.compare(testpattern) == 0)
									{
										//std::cout << "hi" << std::endl;
										flag = 1;
										break;
									}
									else
										flag = 0;
								}
								else
								{
									for(int j = 0; j < 8; j++)
									{
										if(testpattern[j] == 'D')
											continue;
										else if(target[j] == testpattern[j])
										{
											flag = 1;
											//std::cout << "match" << std::endl;
										}
										else
										{
											//std::cout << "breakpoint" << std::endl;
											flag = 0;
											break;
										}
									}
								}

								if(flag == 0)
									continue;
								else
									break;
							}
							//std::cout << flag << std::endl;
							if(flag == 0)
							{
								//std::cout << "reducing" << std::endl;
								img_shrinked.setvalue(i,j,k,1*255);
							}
							else
								img_shrinked.setvalue(i,j,k,img_shrinked.getvalue(i,j,k));
						}


					}

		pixel_count = 0;
		for(int i = 0+1; i < img_shrinked.rows-1; i++)
			for(int j = 0+1; j < img_shrinked.cols-1; j++)
				for(int k = 0; k < img_shrinked.channels; k++)
					if(img_shrinked.getvalue(i,j,k) == 0) pixel_count++;

		if(pixel_count == black_pixel_count) superflag = 0;
		else								 black_pixel_count = pixel_count;
		//std::cout << black_pixel_count << std::endl;


	}
	std::cout << "Total number of ridges: " << black_pixel_count << std::endl;
}

//#############################################################################################

void ImageProc::thin(Image &img_binary, Image &img_thinned)
{
	Image img_after_first("bw", img_binary.rows, img_binary.cols);

		int index  = 0, superflag = 1, flag = 0, black_pixel_count = 0, pixel_count = 0;
		std::string target;
		//char *target_char_arr = new char[9];

		for(int i = 0; i < img_binary.rows; i++)
			for(int j = 0; j < img_binary.cols; j++)
				for(int k = 0; k < img_binary.channels; k++)
					img_thinned.setvalue(i,j,k,img_binary.getvalue(i,j,k));

		for(int i = 0+1; i < img_thinned.rows-1; i++)
			for(int j = 0+1; j < img_thinned.cols-1; j++)
				for(int k = 0; k < img_thinned.channels; k++)
					if(img_thinned.getvalue(i,j,k) == 0) pixel_count++;
		black_pixel_count = pixel_count;

		

		while(superflag == 1)
		{
			for(int i = 0; i < img_thinned.rows; i++)
				for(int j = 0; j < img_thinned.cols; j++)
					for(int k = 0; k < img_thinned.channels; k++)
						img_after_first.setvalue(i,j,k,255);

			for(int i = 0+1; i < img_thinned.rows-1; i++)
				for(int j = 0+1; j < img_thinned.cols-1; j++)
					for(int k = 0; k < img_thinned.channels; k++)
					{
						if(img_thinned.getvalue(i,j,k) == 0)
						{
							index = -1; flag = 0;
							target.clear();
							for(int p = i-1; p <= i+1; p++)
								for(int q = j-1; q <= j+1; q++)
								{
									//std::cout << img_shrinked.getvalue(p,q,k)/255 << std::endl;
									if(!(p==i && q==j))
										if(img_thinned.getvalue(p,q,0) == 0)
											target.append("1");
										else
											target.append("0");
									else
										continue;
								}
							//target_char_arr[8] = NULL;
							//target(target_char_arr);
							//std::cout << "target 1:" << target << std::endl;

							//std::string target_str(target);
							for(int t = 0; t < THIN_FIRST_PATTERNS; t++)
							{
								std::string testpattern(thin_first[t]);
								flag = 0;
								if(testpattern.find('D') == std::string::npos)
								{
									if(target.compare(testpattern) == 0)
									{
										//std::cout << "pattern match" << std::endl;
										flag = 1;
									}
									else
									{
										//std::cout << "pattern nomatch" << std::endl;
										flag = 0;
									}
								}
								else
								{
									for(int j = 0; j < 8; j++)
									{
										if(testpattern[j] == 'D')
											continue;
										else if(target[j] != testpattern[j])
										{
											//std::cout << target << " hey " << testpattern << std::endl;
											flag = 0;
											break;
										}
										else
											flag = 1;
									}
								}

								if(flag == 1)
									break;
								else
									continue;

							}

							if(flag == 1)
								img_after_first.setvalue(i,j,k,0*255);
							else
								img_after_first.setvalue(i,j,k,1*255);

						}
					}

				for(int i = 0+1; i < img_after_first.rows-1; i++)
					for(int j = 0+1; j < img_after_first.cols-1; j++)
						for(int k = 0; k < img_after_first.channels; k++)
						{
							if(img_after_first.getvalue(i,j,k) == 0)
							{
								target.clear();
								index = -1; flag = 0;
								for(int p = i-1; p <= i+1; p++)
									for(int q = j-1; q <= j+1; q++)
									{
										if(!(p==i && q==j))
											if(img_after_first.getvalue(p,q,0) == 0)
												target.append("M");
											else
												target.append("0");
									}

								//std::cout << "target1: " << target;

								for(int t = 0; t < SHRINK_THIN_SECOND_PATTERNS; t++)
								{
									std::string testpattern(shrink__thin_second[t]);
									flag = 1;
									//std::cout << " " << testpattern << std::endl;
									if(testpattern.find('D') == std::string::npos)
									{

										if(target.compare(testpattern) == 0)
										{
											//std::cout << "hi" << std::endl;
											flag = 1;
											break;
										}
										else
											flag = 0;
									}
									else
									{
										for(int j = 0; j < 8; j++)
										{
											if(testpattern[j] == 'D')
												continue;
											else if(target[j] == testpattern[j])
											{
												flag = 1;
												//std::cout << "match" << std::endl;
											}
											else
											{
												//std::cout << "breakpoint" << std::endl;
												flag = 0;
												break;
											}
										}
									}

									if(flag == 0)
										continue;
									else
										break;
								}
								//std::cout << flag << std::endl;
								if(flag == 0)
								{
									//std::cout << "reducing" << std::endl;
									img_thinned.setvalue(i,j,k,1*255);
								}
								else
									img_thinned.setvalue(i,j,k,img_thinned.getvalue(i,j,k));
							}


						}

			pixel_count = 0;
			for(int i = 0+1; i < img_thinned.rows-1; i++)
				for(int j = 0+1; j < img_thinned.cols-1; j++)
					for(int k = 0; k < img_thinned.channels; k++)
						if(img_thinned.getvalue(i,j,k) == 0) pixel_count++;

			if(pixel_count == black_pixel_count) superflag = 0;
			else								 black_pixel_count = pixel_count;
			//std::cout << black_pixel_count << std::endl;


		}

}

//#############################################################################################

void ImageProc::skeletonize(Image &img_binary, Image &img_skeletonized)
{
	Image img_after_first("bw", img_binary.rows, img_binary.cols);

		int index  = 0, superflag = 1, flag = 0, black_pixel_count = 0, pixel_count = 0;
		std::string target;
		//char *target_char_arr = new char[9];

		for(int i = 0; i < img_binary.rows; i++)
			for(int j = 0; j < img_binary.cols; j++)
				for(int k = 0; k < img_binary.channels; k++)
					img_skeletonized.setvalue(i,j,k,img_binary.getvalue(i,j,k));

		for(int i = 0+1; i < img_skeletonized.rows-1; i++)
			for(int j = 0+1; j < img_skeletonized.cols-1; j++)
				for(int k = 0; k < img_skeletonized.channels; k++)
					if(img_skeletonized.getvalue(i,j,k) == 0) pixel_count++;
		black_pixel_count = pixel_count;

		//std::cout << " 1 " << black_pixel_count << std::endl;

		while(superflag == 1)
		{
			for(int i = 0; i < img_skeletonized.rows; i++)
				for(int j = 0; j < img_skeletonized.cols; j++)
					for(int k = 0; k < img_skeletonized.channels; k++)
						img_after_first.setvalue(i,j,k,255);

			for(int i = 0+1; i < img_skeletonized.rows-1; i++)
				for(int j = 0+1; j < img_skeletonized.cols-1; j++)
					for(int k = 0; k < img_skeletonized.channels; k++)
					{
						if(img_skeletonized.getvalue(i,j,k) == 0)
						{
							index = -1; flag = 0;
							target.clear();
							for(int p = i-1; p <= i+1; p++)
								for(int q = j-1; q <= j+1; q++)
								{
									//std::cout << img_shrinked.getvalue(p,q,k)/255 << std::endl;
									if(!(p==i && q==j))
										if(img_skeletonized.getvalue(p,q,0) == 0)
											target.append("1");
										else
											target.append("0");
									else
										continue;
								}
							//target_char_arr[8] = NULL;
							//target(target_char_arr);
							//std::cout << "target 1:" << target << std::endl;

							//std::string target_str(target);
							for(int t = 0; t < SKELETON_FIRST_PATTERNS; t++)
							{
								std::string testpattern(skeleton_first[t]);
								flag = 0;
								if(testpattern.find('D') == std::string::npos)
								{
									if(target.compare(testpattern) == 0)
									{
										//std::cout << "pattern match" << std::endl;
										flag = 1;
									}
									else
									{
										//std::cout << "pattern nomatch" << std::endl;
										flag = 0;
									}
								}
								else
								{
									for(int j = 0; j < 8; j++)
									{
										if(testpattern[j] == 'D')
											continue;
										else if(target[j] != testpattern[j])
										{
											//std::cout << target << " hey " << testpattern << std::endl;
											flag = 0;
											break;
										}
										else
											flag = 1;
									}
								}

								if(flag == 1)
									break;
								else
									continue;

							}

							if(flag == 1)
								img_after_first.setvalue(i,j,k,0*255);
							else
								img_after_first.setvalue(i,j,k,1*255);

						}
					}

				for(int i = 0+1; i < img_after_first.rows-1; i++)
					for(int j = 0+1; j < img_after_first.cols-1; j++)
						for(int k = 0; k < img_after_first.channels; k++)
						{
							if(img_after_first.getvalue(i,j,k) == 0)
							{
								target.clear();
								index = -1; flag = 0;
								for(int p = i-1; p <= i+1; p++)
									for(int q = j-1; q <= j+1; q++)
									{
										if(!(p==i && q==j))
											if(img_after_first.getvalue(p,q,0) == 0)
												target.append("M");
											else
												target.append("0");
									}

								//std::cout << "target1: " << target;

								for(int t = 0; t < SKELETON_SECOND_PATTERNS; t++)
								{
									std::string testpattern(skeleton_second[t]);
									flag = 1;
									//std::cout << " " << testpattern << std::endl;
									if(testpattern.find('D') == std::string::npos)
									{

										if(target.compare(testpattern) == 0)
										{
											//std::cout << "hi" << std::endl;
											flag = 1;
											break;
										}
										else
											flag = 0;
									}
									else
									{
										for(int j = 0; j < 8; j++)
										{
											if(testpattern[j] == 'D')
												continue;
											else if(target[j] == testpattern[j])
											{
												flag = 1;
												//std::cout << "match" << std::endl;
											}
											else
											{
												//std::cout << "breakpoint" << std::endl;
												flag = 0;
												break;
											}
										}
									}

									if(flag == 0)
										continue;
									else
										break;
								}
								//std::cout << flag << std::endl;
								if(flag == 0)
								{
									//std::cout << "reducing" << std::endl;
									img_skeletonized.setvalue(i,j,k,1*255);
								}
								else
									img_skeletonized.setvalue(i,j,k,img_skeletonized.getvalue(i,j,k));
							}


						}

			pixel_count = 0;
			for(int i = 0+1; i < img_skeletonized.rows-1; i++)
				for(int j = 0+1; j < img_skeletonized.cols-1; j++)
					for(int k = 0; k < img_skeletonized.channels; k++)
						if(img_skeletonized.getvalue(i,j,k) == 0) pixel_count++;



			if(pixel_count == black_pixel_count) superflag = 0;
			else								 black_pixel_count = pixel_count;
			//std::cout << black_pixel_count << std::endl;


		}
}

//#############################################################################################

void ImageProc::unsharp_mask(Image &img_orig, Image &img_modified)
{
	int ws = 3;
	double kernel[ws][ws]; //discrete convolution operator

	// //gaussian_kernel(kernel, ws, sigma);
	// double kernel_sum = 0.0;
    int kernel_radius = ws / 2;
 //    //double coeff = 1.0 / (2.0 * M_PI * (sigma * sigma));
 //    double frac = 0.0;

 //    for (int row = -kernel_radius; row <= kernel_radius; row++)
 //        for (int col = -kernel_radius; col <= kernel_radius; col++)
 //        {
 //            frac = exp(-1.0 * ((row*row) + (col*col)) / (2.0 * (sigma*sigma)));

 //            kernel[row + kernel_radius][col + kernel_radius] = frac;

 //            kernel_sum += kernel[row + kernel_radius][col + kernel_radius];
 //        }

 //    for (int i = 0; i < ws; i++)
 //        for (int j = 0; j < ws; j++)
 //            kernel[i][j] = kernel[i][j] / kernel_sum; //Normalized

	kernel[0][0] = -1;	kernel[0][1] = -1;	kernel[0][2] = -1;
	kernel[1][0] = -1;	kernel[1][1] =  8;	kernel[1][2] = -1;
	kernel[2][0] = -1;	kernel[2][1] = -1;	kernel[2][2] = -1;


	for(int i = 0; i < img_orig.rows; i++)
	 	for(int j = 0; j < img_orig.cols; j++)
	 		for(int k = 0; k < img_orig.channels; k++)
		 	{
				int values[ws][ws];
				for(int row = 0; row < ws; row++)
					for(int col = 0; col < ws; col++)
						values[row][col] = 0;
				int values_avg = 0;

				for(int m = -kernel_radius, row = 0; m <= kernel_radius; m++, row++)
					for(int n = -kernel_radius, col = 0; n <= kernel_radius; n++, col++)
					{
						if((i+m >= 0) && (i+m < img_orig.rows) && (j+n >= 0) && (j+n < img_orig.cols))
						{
							values[row][col] = img_orig.getvalue((i+m),(j+n),k) * kernel[row][col];
						}
						// else
						// {
						// 	values[row][col] = img_orig.getvalue((i+m),(j-n),k) * kernel[row][col];
						// }
					}

				for(int row = 0; row < ws; row++)
					for(int col = 0; col < ws; col++)
						values_avg += values[row][col];

				img_modified.setvalue(i,j,k,values_avg);
			}

}

//#############################################################################################
//#############################################################################################

#endif /*__IMAGE_PROC_H__*/
