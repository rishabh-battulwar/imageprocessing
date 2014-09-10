#include <iostream>
#include <cstdlib>
#include <string.h>
#include <ImageProc.h>


void usage(const char *exename)
{
	std::cout << "\n Usage: " << exename << " infile outfile [type] [-w width] [-h height]" << std::endl;
	std::cout << " [options]:" << std::endl;
	std::cout << "\t type :'bw'||'color' ; default: 'color'" << std::endl;
	std::cout << "\t -w width : (default) 512" << std::endl;
	std::cout << "\t -h height : (default) 512\n" << std::endl;
	exit(1);
}

int main(int argc, char* argv[])
{
	int width = 512, height = 512;
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

	ImageProc::color_to_bw(argv[1], argv[2], width, height);
	
	return 0;
}