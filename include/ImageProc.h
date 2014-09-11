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
	std::cout << img_bw(3,3,0) << std::endl;
	std::cout << img_color(35,35,0) <<" "<< img_color(35,35,1) <<" "<< img_color(35,35,2) << std::endl;
	img_bw.imgdata[(3-1)*width*1 + (3-1)*1 + 0] = img_color(3,3,0)*0.21 + img_color(3,3,1)*0.72 + img_color(3,3,2)*0.07;
	std::cout << img_bw(3,3,0) << std::endl;
	std::cout << (int)img_bw.imgdata[(3-1)*width*1 + (3-1)*1 + 0] << std::endl;
	img_color.write_image(outfile, width, height);

	//delete img_color;

}

#endif /*__IMAGE_PROC_H__*/
