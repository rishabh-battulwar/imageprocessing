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
		
};

void ImageProc::color_to_bw(const char *infile, const char *outfile, int width, int height)
{
	//Image* img_color = new Image("color");
	//Image* img_bw = new Image("bw", width, height);
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


#endif /*__IMAGE_PROC_H__*/
