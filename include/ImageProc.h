#ifndef __IMAGE_PROC_H__
#define __IMAGE_PROC_H__

#include <iostream>
#include <math.h>
#include "Image.h"

class ImageProc 
{
	private:

	public:
    	static void color_to_bw(const char *infile, const char *outfile, int width, int height);
		static void image_resize(const char *infile, const char* outfile, int width, int height, int target_width, int target_height);
		static void hist_equal_cumulative(const char *infile, const char *outfile, int width, int height);
		static void hist_equal(const char *infile, const char *outfile, int width, int height);
		static void oil_painting(const char *infile, const char *outfile, int width, int height, int window_size);
		
};

void ImageProc::color_to_bw(const char *infile, const char *outfile, int width, int height)
{
	Image img_color("color", width, height);
	Image img_bw("bw", width, height);

	img_color.read_image(infile, width, height);

	for(int i = 0; i < height; i++)
		for(int j = 0; j < width; j++)
			img_bw.imgdata[(i)*width*1 + (j)*1 + 0] = img_color(i,j,0)*0.21 \
													+ img_color(i,j,1)*0.72 \
													+ img_color(i,j,2)*0.07; //Luminosity method

	img_bw.write_image(outfile, width, height);

	//delete img_color;

}

void ImageProc::image_resize(const char *infile, const char* outfile, int width, int height, int target_width, int target_height)
{
	Image img_orig("color", width, height);
	Image img_resized("color", target_width, target_height);

	img_orig.read_image(infile, width, height);

	float X, Y, x, y, a, b;

	for(int i = 0; i < target_height; i++)
		for(int j = 0; j < target_width; j++)
		{			
			x = height * (1.0 * i / target_height);
			y = width  * (1.0 * j / target_width);

			X = floor(x);
			Y = floor(y);

			//std::cout << height << " " << width << std::endl;
			//std::cout << target_height << " " << target_width << std::endl;
			//std::cout << "x: " << x << " X: " << X << " y: " << y << " Y: " << Y << std::endl;
			a = y - Y;
			b = x - X;

			for(int k = 0; k < 3; k++)
				img_resized.imgdata[(i)*target_width*3 + (j)*3 + k] \
				=  img_orig(X  ,Y  ,k)*(1.0-a)*(1.0-b)\
				  +img_orig(X  ,Y+1,k)*(    a)*(1.0-b)\
				  +img_orig(X+1,Y  ,k)*(1.0-a)*(    b)\
				  +img_orig(X+1,Y+1,k)*(    a)*(    b);
		}

	//img_resized.write_image(outfile, target_width, target_height);
	img_resized.write_image(outfile, target_width, target_height);
}

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
	// 		else    img_bw.imgdata[(i)*width*1 + (j)*1 + 0] = 255 * hist_cumulative[img_bw(i,j,0)] \
	// 														      / hist_cumulative[255];


	// img_bw.write_image(outfile, width, height);

	//######## If image == color ########

	Image img_color("color", width, height);

	img_color.read_image(infile, width, height);

	int histogram_bins[256][3] = {0};

	for(int i = 0; i < height; i++)
		for(int j = 0; j < width; j++)
			for(int k = 0; k < 3; k++)
				if((img_color(i, j, k) >= 0) && (img_color(i, j, k) < 256))
					histogram_bins[img_color(i, j, k)][k]++;


	int hist_cumulative[256][3] = {0};
	for(int k = 0; k < 3; k++)
		hist_cumulative[0][k] = histogram_bins[0][k];
	for(int i = 1; i < 256; i++)
		for(int k = 0; k < 3; k++)
			hist_cumulative[i][k] = hist_cumulative[i-1][k] + histogram_bins[i][k];


	for(int i = 0; i < height; i++)
		for(int j = 0; j < width; j++)
			for(int k = 0; k < 3; k++)
				if      (img_color(i, j, k) < 0  ) img_color.imgdata[(i)*width*3 + (j)*3 + k] = 0  ;
				else if (img_color(i, j, k) > 255) img_color.imgdata[(i)*width*3 + (j)*3 + k] = 255;
				else    img_color.imgdata[(i)*width*3 + (j)*3 + k] = 255 * hist_cumulative[img_color(i,j,k)][k] \
																      / hist_cumulative[255][k];


	img_color.write_image(outfile, width, height);

	//###########

}

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

	//######## If image == color ########

	Image img_color("color", width, height);

	img_color.read_image(infile, width, height);

	int histogram_bins[256][3] = {0};

	for(int i = 0; i < height; i++)
		for(int j = 0; j < width; j++)
			for(int k = 0; k < 3; k++)
				if((img_color(i, j, k) >= 0) && (img_color(i, j, k) < 256))
					histogram_bins[img_color(i, j, k)][k]++;


	int hist_cumulative[256][3] = {0};
	for(int k = 0; k < 3; k++)
		hist_cumulative[0][k] = histogram_bins[0][k];
	for(int i = 1; i < 256; i++)
		for(int k = 0; k < 3; k++)
			hist_cumulative[i][k] = hist_cumulative[i-1][k] + histogram_bins[i][k];

	int hist_cumulative_copy[256][3] = {0};
	int curr_bin_count = 0, prev_bin_count = 0;
	double value =  0;

	for(int i = 0; i < height; i++)
		for(int j = 0; j < width; j++)
			for(int k = 0; k < 3; k++)
				if      (img_color(i, j, k) < 0  ) img_color.imgdata[(i)*width*3 + (j)*3 + k] = 0  ;
				else if (img_color(i, j, k) > 255) img_color.imgdata[(i)*width*3 + (j)*3 + k] = 255;
				else    
				{
					++hist_cumulative_copy[img_color(i, j, 0)][k];
					curr_bin_count = hist_cumulative_copy[img_color(i, j, 0)][k];
					prev_bin_count = hist_cumulative[img_color(i, j, 0) - 1][k];
					value = floor(255.0 * (prev_bin_count + curr_bin_count) / hist_cumulative[255][k]);
					img_color.imgdata[(i)*width*3 + (j)*3 + k] = value;
					value = 0; curr_bin_count = 0; prev_bin_count = 0;
				}


	img_color.write_image(outfile, width, height);

	//###########

}


void ImageProc::oil_painting(const char *infile, const char *outfile, int width, int height, int window_size)
{
	Image img_orig("color", width, height);
	Image img_oil("color", width, height);

	int ws = window_size; //window_size
	int wl = ((-1*ws) + 1) / 2;
	int wr = (( 1*ws) - 1) / 2;
	int values[ws*ws][3];
	int max_occur_val[3], max_occurrence = 0, occurrence = 0;;
	img_orig.read_image(infile, width, height);
	int index;

	for(int i = 0; i < height; i++)
	 	for(int j = 0; j < width; j++)
	 	{
			index = 0;
			max_occur_val[0] = max_occur_val[1] = max_occur_val[2] = 0;
			max_occurrence = 0;
			for(int p = 0; p < ws*ws; p++)
				values[p][0] = values[p][1] = values[p][2] = 0;

			for(int m = wl; m <= wr; m++)
				for(int n = wl; n <= wr; n++)
				{	
					if((i+m >= 0) && (i+m < height) && (j+n >= 0) && (j+n < width))
						for(int k = 0; k < 3; k++)
						{
							values[index][k] = img_orig(i+m, j+n, k);
						}
					else
						for(int k = 0; k < 3; k++)
							values[index][k] = 280;

					index++;

				}

			for(int p = 0; p < ws*ws; p++)
			{	
				if(!(values[p][0] == 280) && !(values[p][1] == 280) && !(values[p][0] == 280))
				{
					occurrence = 0;
					for(int r = 0; r < ws*ws; r++)
					{
						if(values[p][0] == values[r][0] &&
						   values[p][1] == values[r][1] &&
						   values[p][2] == values[r][2]) occurrence++;

					}

					if(max_occurrence < occurrence) 
					{
						max_occurrence = occurrence;
						max_occur_val[0] = values[p][0];
						max_occur_val[1] = values[p][1];
						max_occur_val[2] = values[p][2];
					}
				}
			}

			img_oil.imgdata[i*width*3 + j*3 + 0] = max_occur_val[0];
			img_oil.imgdata[i*width*3 + j*3 + 1] = max_occur_val[1];
			img_oil.imgdata[i*width*3 + j*3 + 2] = max_occur_val[2];
			
			
	 	}

	img_oil.write_image(outfile, width, height);
}


#endif /*__IMAGE_PROC_H__*/
