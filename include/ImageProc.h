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
		static void oil_paint_filter(Image &img_orig, Image &img_modified, int window_size);
		static void gaussian_filter(Image &img_orig, Image &img_modified, int window_size);
		static void apply_gaussian_filter(const char *infile, const char *outfile, int width, int height, int window_size);
		
};

void ImageProc::color_to_bw(const char *infile, const char *outfile, int width, int height)
{
	Image img_color("color", width, height);
	Image img_bw("bw", width, height);

	img_color.read_image(infile, img_color.cols, img_color.rows);

	for(int i = 0; i < img_bw.rows; i++)
		for(int j = 0; j < img_bw.cols; j++)
			for(int k = 0; k < img_bw.channels; k++)
				img_bw.setvalue(i,j,k,((img_color.getvalue(i,j,0)*0.21)\
									  +(img_color.getvalue(i,j,1)*0.72)\
									  +(img_color.getvalue(i,j,2)*0.07)));

	img_bw.write_image(outfile, img_bw.cols, img_bw.rows);

}

void ImageProc::image_resize(const char *infile, const char* outfile, int width, int height, int target_width, int target_height)
{
	Image img_orig("color", width, height);
	Image img_resized("color", target_width, target_height);

	img_orig.read_image(infile, img_orig.cols, img_orig.rows);

	float X, Y, x, y, a, b;

	for(int i = 0; i < img_resized.rows; i++)
		for(int j = 0; j < img_resized.cols; j++)
		{			
			x = img_orig.rows * (1.0 * i / img_resized.rows);
			y = img_orig.cols * (1.0 * j / img_resized.cols);

			X = floor(x);
			Y = floor(y);

			//std::cout << height << " " << width << std::endl;
			//std::cout << target_height << " " << target_width << std::endl;
			//std::cout << "x: " << x << " X: " << X << " y: " << y << " Y: " << Y << std::endl;
			a = y - Y;
			b = x - X;

			for(int k = 0; k < img_resized.channels; k++)
				img_resized.setvalue(i,j,k,((img_orig.getvalue(X  ,Y  ,k)*(1.0-a)*(1.0-b))
				  						   +(img_orig.getvalue(X  ,Y+1,k)*(    a)*(1.0-b))
				  						   +(img_orig.getvalue(X+1,Y  ,k)*(1.0-a)*(    b))
				  						   +(img_orig.getvalue(X+1,Y+1,k)*(    a)*(    b))
				  						   )
									);
		}

	img_resized.write_image(outfile, img_resized.cols, img_resized.rows);
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

	img_orig.read_image(infile, img_orig.cols, img_orig.rows);
	oil_paint_filter(img_orig, img_oil, window_size);

	img_oil.write_image(outfile, img_oil.cols, img_oil.rows);
}


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

void ImageProc::apply_gaussian_filter(const char *infile, const char *outfile, int width, int height, int window_size)
{
	Image img_orig("bw", width, height);
	Image img_modified("bw", width, height);

	img_orig.read_image(infile, img_orig.cols, img_orig.rows);
	//oil_paint_filter(img_orig, img_oil, window_size);
	gaussian_filter(img_orig, img_modified, window_size);

	img_modified.write_image(outfile, img_modified.cols, img_modified.rows);
}

void ImageProc::gaussian_filter(Image &img_orig, Image &img_modified, int window_size)
{
	int ws = window_size; //window_size
	int wleft = ((-1*ws) + 1) / 2;
	int wright = (( 1*ws) - 1) / 2;
	int wtop = wleft;
	int wbottom = wright;

	double gaussian_kernel[ws][ws], kernel_sum = 0, normalized_gaussian_kernel[ws][ws];
	double row_exp = 0, col_exp;
	for(int i = wtop, row = 0; i <= wbottom; i++, row++)
	{
		col_exp = row_exp;

		for(int j = wleft, col = 0; j <= wright; j++, col++)
		{	
			gaussian_kernel[row][col] = pow(2.0, col_exp);

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
			kernel_sum += gaussian_kernel[row][col];

	for(int row = 0; row < ws; row++)
		for(int col = 0; col < ws; col++)
			normalized_gaussian_kernel[row][col] = gaussian_kernel[row][col] / kernel_sum;

	// for(int row = 0; row < ws; row++)
	// {
	// 	std::cout << " " << std::endl;
	// 	for(int col = 0; col < ws; col++)
	// 		std::cout << normalized_gaussian_kernel[row][col] << " " ;
	// }


	for(int i = 0; i < img_orig.rows; i++)
	 	for(int j = 0; j < img_orig.cols; j++)
	 		for(int k = 0; k < img_orig.channels; k++)
		 	{
				//max_occur_val[0] = max_occur_val[1] = max_occur_val[2] = 0;
				//max_occurrence = 0;
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
							values[row][col] = img_orig.getvalue((i+m),(j+n),k) * normalized_gaussian_kernel[row][col];
						}
						else
						{
							values[row][col] = img_orig.getvalue((i+m),(j-n),k) * normalized_gaussian_kernel[row][col];
						}
					}

				for(int row = 0; row < ws; row++)
					for(int col = 0; col < ws; col++)
						values_avg += values[row][col];

				img_modified.setvalue(i,j,k,values_avg);
			}

}


#endif /*__IMAGE_PROC_H__*/
