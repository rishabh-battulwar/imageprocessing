#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


class Image
{
  private:
    //Image Data
    char *imgtype;
    int cols, rows;
    int channels;

  public:
    unsigned char *imgdata;

    Image(char *type);
    Image(char *type, int width, int height);
    ~Image() {};
    void read_image(const char *filename, int width, int height);
    void write_image(const char *filename, int width, int height);
    int operator()(int K, int L, int M);
    
};

Image::Image(char *type)
{
  if(!strcmp(type, "bw")) 
  {
    imgtype = "bw";
    channels = 1;
    cols = 512; //default
    rows = 512; //default
    imgdata = new unsigned char [cols*rows*channels];
  }
  else
  {
    imgtype = "color";
    channels = 3;
    cols = 512; //default
    rows = 512; //default
    imgdata = new unsigned char [cols*rows*channels];
  }
}

Image::Image(char *type, int width, int height)
{
  if(!strcmp(type, "bw")) 
  {
    imgtype = "bw";
    channels = 1;
    cols = width; //default
    rows = height; //default
    imgdata = new unsigned char [cols*rows*channels];
  }
  else
  {
    imgtype = "color";
    channels = 3;
    cols = width; //default
    rows = height; //default
    imgdata = new unsigned char [cols*rows*channels];
  }
}


void Image::read_image(const char *filename, int width, int height)
{
  if (!strcmp(imgtype, "color"))
  {
    cols = width;
    rows = height;

    FILE *file;
    if (!(file=fopen(filename,"rb"))) 
    {
      std::cout << "Cannot open file: " << filename << std::endl;
      exit(1);
    }
    if (!(fread(imgdata, sizeof(int), cols*rows*channels, file)))
      std::cout << "Error Reading" << std::endl;
    fclose(file);  
    //std::cout << "This is the data " << (0,41,0) << " ends here" << std::endl;
  }

  else if (!strcmp(imgtype, "bw"))
  {
    cols = width;
    rows = height;

    FILE *file;
    if (!(file=fopen(filename,"rb"))) 
    {
      std::cout << "Cannot open file: " << filename << std::endl;
      exit(1);
    }
    if (!(fread(imgdata, sizeof(int), cols*rows*channels, file)))
      std::cout << "Error Reading" << std::endl;
    fclose(file);  
    
  }

  else
  {
    std::cout << "Filetype wasn't right!" << std::endl;
  }

}

void Image::write_image(const char *filename, int width, int height)
{
  if (!strcmp(imgtype, "color"))
  {
    cols = width;
    rows = height;

    FILE *file;
    if (!(file=fopen(filename,"wb"))) 
    {
      std::cout << "Cannot open file: " << filename << std::endl;
      exit(1);
    }
    if (!(fwrite(imgdata, sizeof(unsigned char), cols*rows*channels, file)))
       std::cout << "Error Writing" << std::endl;
    fclose(file);
    
    // for(int i = 0; i < size; i++)
    // {
    //   for(int j = 0; j < size; j++)
    //   {
    //     delete [] imgdata[i][j];
    //   }
    //   delete [] imgdata[i];
    // }
    // delete [] imgdata;

  }

  else if (!strcmp(imgtype, "bw"))
  {
    cols = width;
    rows = height;

    FILE *file;

    if (!(file=fopen(filename,"wb"))) 
    {
      std::cout << "Cannot open file: " << filename << std::endl;
      exit(1);
    }
    if (!(fwrite(imgdata, sizeof(unsigned char), cols*rows*channels, file)))
      std::cout << "Error Writing" << std::endl;
    fclose(file);  

  }

}

int Image::operator()(int K, int L, int M)
    {
    	if(!(strcmp(imgtype, "color")))
      	return (int)imgdata[(K-1)*cols*channels + (L-1)*channels + M];
      if(!(strcmp(imgtype, "bw")))
      	return (int)imgdata[(K-1)*cols*channels + (L-1)*channels + M];
      else
        return 1;
    }





#endif /*__IMAGE_H__*/
