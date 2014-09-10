#ifndef __IMAGE_PROC_H__
#define __IMAGE_PROC_H__

#include <iostream>
#include "Image.h"

class ImageProc 
{
	private:

	public:
    	static void color_to_bw(const char *infile, const char *outfile, int width, int height);
		static void image_resize(const char *, int cols, int rows);
		
};

void ImageProc::color_to_bw(const char *infile, const char *outfile, int width, int height)
{
	//Image* img_color = new Image("color");
	//Image* img_bw = new Image("bw", width, height);
	Image img_color("color");
	Image img_bw("bw", width, height);

	img_color.read_image(infile, width, height);

	// for(int i = 0; i < width; i++)
	// 	for(int j = 0; j < height; j++)
	// 		img_bw.imgdata[(i*width+j)*height+0] = img_color(i,j,0)*0.21 + img_color(i,j,1)*0.72 + img_color(i,j,2)*0.07;

	img_color.write_image(outfile, width, height);

	//delete img_color;

}

#endif /*__IMAGE_PROC_H__*/
