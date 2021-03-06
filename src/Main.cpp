/*///////////////////////////////
# EE569 Homework Assignment #1
# Date: Sept 21, 2014
# Name: Rishabh Battulwar
# ID: 4438-1435-20
# email: battulwa@usc.edu
#
# Compiled on CYGWIN with g++
*/////////////////////////////////

#include <iostream>
#include <cstdlib>
#include <string.h>
#include <ImageProc.h>


void usage(const char *exename)
{
	std::cout << "\n General Usage: " << exename << " -prob {method_name} infile outfile [type] [-w width] [-h height] [-tw target_width] [-th target_height] [-ws window_size]" << std::endl;
	std::cout << " [Specific Usage]: \n" << std::endl;
	
	std::cout << " Problem 1.a1. Usage: " << exename << " -prob sobel -i <input file path> -o <output file path> [-w 'width'] [-h 'height']  [-perc 'percentage']" << std::endl;
	std::cout << "     -----> Example: " << exename << " -prob sobel -i ../images/elaine.raw -o ../output_images/elaine_edge_map.raw -w 256 -h 256 -perc 10" << "\n" << std::endl;
	
	std::cout << " Problem 1.a2. Usage: " << exename << "  -prob LoG -i <input file path> -o <output file path> [-w 'width'] [-h 'height']" << std::endl;
	std::cout << "     -----> Example: " << exename << " -prob LoG -i ../images/elaine.raw -o ../output_images/elaine_LoG_edge_map.raw -w 256 -h 256" << "\n" << std::endl;
	
	std::cout << " Problem 1.b. Usage: " << exename << " -prob art_effect -i <input file path> -o <output file path> [-w 'width'] [-h 'height']" << std::endl;
	std::cout << "     -----> Example: " << exename << " -prob art_effect -i ../images/Scarlett.raw -o ../output_images/Scarlett_out.raw -w 400 -h 300" << "\n" <<std::endl;
	
	std::cout << " Problem 2.1. Usage: " << exename << " -prob binary -i <input file path> -o <output file path> [-w 'width'] [-h 'height']" << std::endl;
	std::cout << "     -----> Example: " << exename << " -prob binary -i ../images/fingerprint_good.raw -o ../output_images/fingerprint_good_binary.raw -w 388 -h 374 " << "\n" << std::endl;
	
	std::cout << " Problem 2.2. Usage: " << exename << " -prob fingerprint -i <input file path> -o <output file path> [-w 'width'] [-h 'height']" << std::endl;
	std::cout << "     -----> Example: " << exename << " -prob fingerprint -i ../images/fingerprint_good.raw -o ../output_images/fingerprint_good_processed.raw -w 388 -h 374" << "\n" << std::endl;
	
	std::cout << " Problem 2.3. Usage: " << exename << " -prob minutiae -i <input file path> -o <output file path> [-w 'width'] [-h 'height'] " << std::endl;
	std::cout << "     -----> Example: " << exename << " -prob minutiae -i ../images/fingerprint_good.raw -o ../output_images/fingerprint_good_extraction.raw -w 388 -h 374" << "\n" << std::endl;
	
	std::cout << " Problem 3.1. Usage: " << exename << " -prob halftone -i <input file path> [-w 'width'] [-h 'height']" << std::endl;
	std::cout << "     -----> Example: " << exename << " -prob halftone -i ../images/boat.raw -w 512 -h 512" << "\n" << std::endl;
	
	std::cout << " Problem 3.2. Usage: " << exename << " -prob color_halftone -i <input file path> [-w 'width'] [-h 'height']" << std::endl;
	std::cout << "     -----> Example: " << exename << " -prob color_halftone -i ../images/trees.raw -o ../output_images/color_halftone_trees.raw -w 350 -h 258" << "\n" << std::endl;
	
	// std::cout << " Problem 1.a. Usage: " << exename << " -prob color_to_bw -i <input file path> -o <output file path> [-w 'width'] [-h 'height']" << std::endl;
	// std::cout << "     -----> Example: " << exename << " -prob color_to_bw -i ../images/mandril.raw -o ../images/mandril_bw.raw -w 512 -h 512" << "\n" << std::endl;
	
	// std::cout << " Problem 1.b. Usage: " << exename << " -prob image_resize -i <input file path> -o <output file path> [-w 'width'] [-h 'height'] [-tw 'target_width'] [-th 'target_height]" << std::endl;
	// std::cout << "     -----> Example: " << exename << " -prob image_resize -i ../images/pepper.raw -o ../images/pepper_scale.raw -w 512 -h 512 -tw 700 -th 700" << "\n" << std::endl;
	
	// std::cout << " Problem 2.a1 Usage: " << exename << " -prob hist_eq_cum -i <input file path> -o <output file path> [-w 'width'] [-h 'height']" << std::endl;
	// std::cout << "     -----> Example: " << exename << " -prob hist_eq_cum -i ../images/Girl.raw -o ../images/Girl_hist_eq_cum.raw -w 256 -h 256" << "\n" <<std::endl;
	
	// std::cout << " Problem 2.a2 Usage: " << exename << " -prob hist_equal -i <input file path> -o <output file path> [-w 'width'] [-h 'height']" << std::endl;
	// std::cout << "     -----> Example: " << exename << " -prob hist_equal -i ../images/Girl.raw -o ../images/Girl_hist_eq.raw -w 256 -h 256" << "\n" << std::endl;
	
	// std::cout << " Problem 2.b. Usage: " << exename << " -prob oil_paint_effect -i <input file path> -o <output file path> [-w 'width'] [-h 'height'] [-ws 'window_size']" << std::endl;
	// std::cout << "     -----> Example: " << exename << " -prob oil_paint_effect -i ../images/Trojan_256.raw -o ../images/Trojan_256_oil.raw -w 384 -h 384 -ws 3" << "\n" << std::endl;
	
	// std::cout << " Problem 2.c. Usage: " << exename << " -prob special_effect -i <input file path> -o <output file path> [-w 'width'] [-h 'height']" << std::endl;
	// std::cout << "     -----> Example: " << exename << " -prob special_effect -i ../images/chat.raw -o ../images/chat_effect.raw -w 481 -h 321" << "\n" << std::endl;
	
	// std::cout << " Problem 3.a. Usage: " << exename << " -prob denoising -i <input file path> -o <output file path> [-w 'width'] [-h 'height'] [-sigma 'value'] [-thresh 'threshold_value']" << std::endl;
	// std::cout << "     -----> Example: " << exename << " -prob denoising -i ../images/Lena_mixed.raw -o ../images/Lena_clean.raw -w 512 -h 512 -sigma 3 -thresh 20.0" << "\n" << std::endl;
	
	// std::cout << " Problem 3.b. Usage: " << exename << " -prob bilateral_filter -i <input file path> -o <output file path> [-w 'width'] [-h 'height'] [-tw 'target_width'] [-th 'target_height] [-sigma 'value'] [-sigma_sim 'sigma_similarity'] [-iter '#iterations']" << std::endl;
	// std::cout << "     -----> Example: " << exename << " -prob bilateral_filter -i ../images/pepper.raw -o ../images/pepper_scale.raw -w 512 -h 512 -ws 3 -sigma 2 -sigma_sim 20 -iter 5" << "\n" << std::endl;
	
	// std::cout << " Problem 3.c. Usage: " << exename << " -prob non_local_mean -i <input file path> -o <output file path> [-w 'width'] [-h 'height'] [-tw 'target_width'] [-th 'target_height] [-rs 'region_size'] [-sigma 'value'] [-h 'h_value']" << std::endl;
	// std::cout << "     -----> Example: " << exename << " -prob non_local_mean -i ../images/pepper.raw -o ../images/pepper_scale.raw -w 512 -h 512 -rs 5 -sigma 3 -h 5" << "\n" << std::endl;

	std::cout << "\t type :'bw'||'color' ; default: 'color'" << std::endl;
	std::cout << "\t -w width : (default) 512" 				<< std::endl;
	std::cout << "\t -h height : (default) 512" 			<< std::endl;
	std::cout << "\t -tw target_width : (default) 700" 		<< std::endl;
	std::cout << "\t -th target_height : (default) 700\n" 	<< std::endl;
	std::cout << "\t -th target_height : (default) 700\n" 	<< std::endl;
	exit(1);
}

int main(int argc, char* argv[])
{
	int width = 512, height = 512;
	int window_size = 3, region_size = 5;
	int sigma = 1, h = 1, sigma_sim = 20;
	float threshold = 20.0, tau = 0.99, epsilon = 0.3, phi = 2;
	int target_width = 700, target_height = 700;
	int iter = 6, percentage = 10;
	//char *type = "color";
	char *method_name = "color_to_bw";
	char *infile = "../images/mandril.raw";
	char *outfile = "../images/mandril_bw.raw";
	if (argc < 3)
		usage(argv[0]);

	for(int i = 1; i < argc; i++)
	{
		if     (!(strcmp(argv[i], "-prob"))) method_name = argv[++i];
		else if(!(strcmp(argv[i], "-i"))) infile = argv[++i];
		else if(!(strcmp(argv[i], "-o"))) outfile = argv[++i];
		//else if(!(strcmp(argv[i], "bw")) || !(strcmp(argv[i], "color"))) type = argv[i];
		else if(!(strcmp(argv[i], "-w"))) width = atoi(argv[++i]);
		else if(!(strcmp(argv[i], "-h"))) height = atoi(argv[++i]);
		else if(!(strcmp(argv[i], "-tw"))) target_width = atoi(argv[++i]);
		else if(!(strcmp(argv[i], "-th"))) target_height = atoi(argv[++i]);
		else if(!(strcmp(argv[i], "-ws"))) window_size = atoi(argv[++i]);
		else if(!(strcmp(argv[i], "-rs"))) region_size = atoi(argv[++i]);
		else if(!(strcmp(argv[i], "-sigma"))) sigma = atoi(argv[++i]);
		else if(!(strcmp(argv[i], "-sigma_sim"))) sigma_sim = atoi(argv[++i]);
		else if(!(strcmp(argv[i], "-iter"))) iter = atoi(argv[++i]);
		else if(!(strcmp(argv[i], "-perc"))) percentage = atoi(argv[++i]);
		else if(!(strcmp(argv[i], "-h"))) h = atoi(argv[++i]);
		else if(!(strcmp(argv[i], "-thresh"))) threshold = atof(argv[++i]);
		else if(!(strcmp(argv[i], "-tau"))) tau = atof(argv[++i]);
		else if(!(strcmp(argv[i], "-epsilon"))) epsilon = atof(argv[++i]);
		else if(!(strcmp(argv[i], "-phi"))) phi = atof(argv[++i]);
		else
		{
			std::cerr << "\n Unknown Option : " << argv[i] << std::endl;
			usage(argv[0]);
		}
	}


	if     (!(strcmp(method_name, "color_to_bw"))) 			ImageProc::color_to_bw(infile, outfile, width, height);
	else if(!(strcmp(method_name, "image_resize")))			ImageProc::image_resize(infile, outfile, width, height, target_width, target_height);
	else if(!(strcmp(method_name, "hist_eq_cum")))			ImageProc::hist_equal_cumulative(infile, outfile, width, height);
	else if(!(strcmp(method_name, "hist_equal")))			ImageProc::hist_equal(infile, outfile, width, height);
	else if(!(strcmp(method_name, "oil_paint_effect")))		ImageProc::oil_painting(infile, outfile, width, height, window_size);
	else if(!(strcmp(method_name, "special_effect")))		ImageProc::film_special_effect(infile, outfile, width, height);
	else if(!(strcmp(method_name, "denoising")))			ImageProc::denoising(infile, outfile, width, height, window_size, sigma, threshold);
	else if(!(strcmp(method_name, "bilateral_filter")))		ImageProc::apply_bilateral_filter(infile, outfile, width, height, window_size, sigma, sigma_sim, iter);
	else if(!(strcmp(method_name, "non_local_mean")))		ImageProc::apply_non_local_mean(infile, outfile, width, height, window_size, region_size, sigma, h);
	else if(!(strcmp(method_name, "sobel")))				ImageProc::apply_sobel_operator(infile, outfile, width, height, percentage);
	else if(!(strcmp(method_name, "LoG")))					ImageProc::apply_LoG_operator(infile, outfile, width, height);
	else if(!(strcmp(method_name, "halftone")))				ImageProc::halftoning(infile, width, height);
	else if(!(strcmp(method_name, "color_halftone")))		ImageProc::color_halftoning(infile, outfile, width, height);
	else if(!(strcmp(method_name, "art_effect")))			ImageProc::apply_artistic_effect(infile, outfile, width, height, tau, epsilon, phi, window_size, sigma, sigma_sim, iter);
	else if(!(strcmp(method_name, "fingerprint")))			ImageProc::apply_morphology(infile, outfile, width, height);
	else if(!(strcmp(method_name, "binary")))				ImageProc::apply_binarization(infile, outfile, width, height);
	else if(!(strcmp(method_name, "minutiae")))				ImageProc::apply_minutae_extraction(infile, outfile, width, height);

	else
	{
		std::cout << "method_name doesn't match any method!" << std::endl;
		usage(argv[0]);
	}
	return 0;
}