#include <iostream>
#include <cstdlib>
#include <string.h>
#include <ImageProc.h>


void usage(const char *exename)
{
	std::cout << "\n Usage: " << exename << " infile outfile [type] [-w width] [-h height] [-tw target_width] [-th target_height]" << std::endl;
	std::cout << " [options]:" << std::endl;
	std::cout << "\t type :'bw'||'color' ; default: 'color'" << std::endl;
	std::cout << "\t -w width : (default) 512" 				<< std::endl;
	std::cout << "\t -h height : (default) 512" 			<< std::endl;
	std::cout << "\t -tw target_width : (default) 700" 		<< std::endl;
	std::cout << "\t -th target_height : (default) 700\n" 	<< std::endl;
	exit(1);
}

int main(int argc, char* argv[])
{
	int width = 512, height = 512;
	int target_width = 700, target_height = 700;
	char *type = "color";
	if (argc < 3)
		usage(argv[0]);

	for(int i = 3; i < argc; i++)
	{
		if(!(strcmp(argv[i], "bw")) || !(strcmp(argv[i], "color"))) type = argv[i];
		else if(!(strcmp(argv[i], "-w"))) width = atoi(argv[++i]);
		else if(!(strcmp(argv[i], "-h"))) height = atoi(argv[++i]);
		else 
		{
			std::cerr << "\n Unknown Option : " << argv[i] << std::endl;
			usage(argv[0]);
		}
	}

	//ImageProc::color_to_bw(argv[1], argv[2], width, height);
	ImageProc::image_resize(argv[1], argv[2], width, height, target_width, target_height);
	
	return 0;
}