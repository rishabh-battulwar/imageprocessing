/*///////////////////////////////
# EE569 Homework Assignment #1
# Date: Sept 21, 2014
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
#include "Image.h"

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
		static void gaussian_filter(Image &img_orig, Image &img_modified, int window_size, int sigma);
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

		static void apply_sobel_operator(const char *infile, const char *outfile, int width, int height);
		static void sobel_edge_detector(Image &img_orig, Image &img_modified);
		static int func_gradient(int **operator_mask, Image &img_orig, int x, int y, int z, int window_size);
		static void apply_LoG_operator(const char *infile, const char *outfile, int width, int height);
		static void LoG_edge_detector(Image &img_orig, Image &img_modified);
		static int check_zero_crossing(Image &img_modified_copy, int i, int j, int k);
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

void ImageProc::gaussian_filter(Image &img_orig, Image &img_modified, int window_size, int sigma)
{
	int ws = window_size;
	double kernel[ws][ws]; //discrete convolution operator

	//gaussian_kernel(kernel, ws, sigma);
	double kernel_sum = 0.0; 
    int kernel_radius = ws / 2; 
    //double coeff = 1.0 / (2.0 * M_PI * (sigma * sigma)); 
    double frac = 0.0; 

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
						else
						{
							values[row][col] = img_orig.getvalue((i+m),(j-n),k) * kernel[row][col];
						}
					}

				for(int row = 0; row < ws; row++)
					for(int col = 0; col < ws; col++)
						values_avg += values[row][col];

				img_modified.setvalue(i,j,k,values_avg);
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


void ImageProc::apply_sobel_operator(const char *infile, const char *outfile, int width, int height)
{
	Image img_orig("bw", width, height);
	Image img_modified("bw", width, height);
	img_orig.read_image(infile, img_orig.cols, img_orig.rows);
	
	sobel_edge_detector(img_orig, img_modified);
	std::cout << img_modified.getvalue(1,1,0) << std::endl;
	img_modified.write_image(outfile, img_modified.cols, img_modified.rows);
}

void ImageProc::sobel_edge_detector(Image &img_orig, Image &img_modified)
{
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
	int sum = 0;
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

					//sum = abs(Gx[i][j]) + abs(Gy[i][j]);
					//sum = sum > 255 ? 255 : sum;
					//sum = sum < 0   ? 0   : sum;
					//img_modified.setvalue(i,j,k,sum);
				}
				
			}

	//Normalize Gradient values
	int abs_minGx = abs(minGx);
	int abs_minGy = abs(minGy);
	maxGx = maxGx + abs_minGx;
	maxGy = maxGy + abs_minGy;
	std::cout << minGx << " " << maxGx << std::endl;

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			Gx[i][j] = Gx[i][j] + abs_minGx;
			Gx[i][j] = (int)(1.0*Gx[i][j]*255/maxGx);
			Gy[i][j] = Gy[i][j] + abs_minGy;
			Gy[i][j] = (int)(1.0*Gy[i][j]*255/maxGy);
		}

	int nGradient;
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
			{
				if((i-1)>=0 && (i+1)<img_orig.rows && (j-1)>=0 && (j+1)<img_orig.cols)
				{
					//nGradient = (int)sqrt(Gx[i][j]*Gx[i][j] + Gy[i][j]*Gy[i][j]);
					//img_modified.setvalue(i,j,k,nGradient);
					img_modified.setvalue(i,j,k,Gx[i][j]);
					// if(nGradient < 50 || (i == 135 && j == 133))
					// {
					// 	std::cout << i << " " << j << " " << Gx[i][j] << " " << Gy[i][j] << " " << nGradient << " " << img_modified.getvalue(i,j,k) << std::endl;
					// }
				}
				
			}

	//std::cout << "Hey Man " << Gx[127][127] << std::endl;
	Image img_modified_copy = img_modified;
	int histogram_bins[256], hist_cumulative[256];
	get_histogram(img_modified, histogram_bins);
	get_cumulative_histogram(histogram_bins, hist_cumulative);

	int threshold = (int)(0.9*hist_cumulative[255]);
	std::cout << threshold << " " << hist_cumulative[255] << std::endl;

	for(int i = 0; i < 256; i++)
	{
		if(hist_cumulative[i] > threshold)
		{
			threshold = i-1;
			break;
		}
	}
	std::cout << threshold << std::endl;
	for(int i = 0; i < img_modified.rows; i++)
		for(int j = 0; j < img_modified.cols; j++)
			for(int k = 0; k < img_modified.channels; k++)
			{
				if((i-1)>=0 && (i+1)<img_modified.rows && (j-1)>=0 && (j+1)<img_modified.cols)
				{
					if(img_modified.getvalue(i,j,k) < threshold)
						img_modified.setvalue(i,j,k,0);
					else
						img_modified.setvalue(i,j,k,255);
					//nGradient = (int)sqrt(Gx[i][j]*Gx[i][j] + Gy[i][j]*Gy[i][j]);
					//img_modified.setvalue(i,j,k,nGradient);
					// if(nGradient < 50 || (i == 210 && j == 239))
					// {
					// 	std::cout << i << " " << j << " " << Gx[i][j] << " " << Gy[i][j] << " " << nGradient << " " << img_modified.getvalue(i,j,k) << std::endl;
					// }
				}
			}
}

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

void ImageProc::get_cumulative_histogram(int histogram_bins[256], int hist_cumulative[256])
{
	hist_cumulative[0] = histogram_bins[0];
	for(int i = 1; i < 256; i++)
		hist_cumulative[i] = hist_cumulative[i-1] + histogram_bins[i];
}


void ImageProc::apply_LoG_operator(const char *infile, const char *outfile, int width, int height)
{
	Image img_orig("bw", width, height);
	Image img_modified("bw", width, height);
	img_orig.read_image(infile, img_orig.cols, img_orig.rows);
	
	LoG_edge_detector(img_orig, img_modified);
	//std::cout << img_modified.getvalue(1,1,0) << std::endl;
	img_modified.write_image(outfile, img_modified.cols, img_modified.rows);

}

void ImageProc::LoG_edge_detector(Image &img_orig, Image &img_modified)
{
	int **LoG_kernel;
	LoG_kernel = new int *[3];
	for(int i = 0; i < 3; i++)
		LoG_kernel[i] = new int[3];

	LoG_kernel[0][0] =  0;	LoG_kernel[0][1] = -1;	LoG_kernel[0][2] =  0;
	LoG_kernel[1][0] = -1;	LoG_kernel[1][1] =  4;	LoG_kernel[1][2] = -1;
	LoG_kernel[2][0] =  0;	LoG_kernel[2][1] = -1;	LoG_kernel[2][2] =  0;

	int Gradient_LoG[img_orig.rows][img_orig.cols];
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			Gradient_LoG[i][j] = 0;
		}
	int minGradient = std::numeric_limits<int>::max();
	int maxGradient = std::numeric_limits<int>::min();
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
			{
				if((i-1)>=0 && (i+1)<img_orig.rows && (j-1)>=0 && (j+1)<img_orig.cols)
				{
					Gradient_LoG[i][j] = func_gradient(LoG_kernel, img_orig, i, j, k, 3);
					if(Gradient_LoG[i][j] < minGradient) minGradient = Gradient_LoG[i][j];
					if(Gradient_LoG[i][j] > maxGradient) maxGradient = Gradient_LoG[i][j];
					
					
					//sum = abs(Gx[i][j]) + abs(Gy[i][j]);
					//sum = sum > 255 ? 255 : sum;
					//sum = sum < 0   ? 0   : sum;
					//img_modified.setvalue(i,j,k,sum);
				}
				
			}

	//Normalize Gradient values
	int abs_minGradient = abs(minGradient);
	maxGradient = maxGradient + abs_minGradient;
	//std::cout << minGx << " " << maxGx << std::endl;

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
		{
			Gradient_LoG[i][j] = Gradient_LoG[i][j] + abs_minGradient;
			Gradient_LoG[i][j] = (int)(1.0*Gradient_LoG[i][j]*255/maxGradient);
		}

	//int nGradient;
	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
			{
				if((i-1)>=0 && (i+1)<img_orig.rows && (j-1)>=0 && (j+1)<img_orig.cols)
				{
					//nGradient = (int)sqrt(Gx[i][j]*Gx[i][j] + Gy[i][j]*Gy[i][j]);
					//img_modified.setvalue(i,j,k,nGradient);
					img_modified.setvalue(i,j,k,Gradient_LoG[i][j]);
					// if(nGradient < 50 || (i == 135 && j == 133))
					// {
					// 	std::cout << i << " " << j << " " << Gx[i][j] << " " << Gy[i][j] << " " << nGradient << " " << img_modified.getvalue(i,j,k) << std::endl;
					// }
				}
				
			}

	for(int i = 0; i < img_orig.rows; i++)
		for(int j = 0; j < img_orig.cols; j++)
			for(int k = 0; k < img_orig.channels; k++)
			{
				if((i-1)>=0 && (i+1)<img_orig.rows && (j-1)>=0 && (j+1)<img_orig.cols)
				{
					if      (img_modified.getvalue(i,j,k) < 100)	img_modified.setvalue(i,j,k,64);
					else if (img_modified.getvalue(i,j,k) > 130)	img_modified.setvalue(i,j,k,192);
					else 											img_modified.setvalue(i,j,k,128);
				}
				
			}

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

#endif /*__IMAGE_PROC_H__*/
