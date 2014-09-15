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
    

  public:
    char *imgtype;
    int cols, rows;
    int channels;
    unsigned char *imgdata;

    Image(char *type);
    Image(char *type, int width, int height);
    ~Image() {};
    void read_image(const char *filename, int width, int height);
    void write_image(const char *filename, int width, int height);
    int getvalue(int i, int j, int k);
    void setvalue(int i, int j, int k, int value);
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
    //cols = width;
    //rows = height;

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
    // cols = width;
    // rows = height;

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
    // cols = width;
    // rows = height;

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

  else if (!strcmp(imgtype, "bw"))
  {
    // cols = width;
    // rows = height;

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
  	return (int)imgdata[(K)*cols*channels + (L)*channels + M];
  if(!(strcmp(imgtype, "bw")))
  	return (int)imgdata[(K)*cols*channels + (L)*channels + M];
  else
    return 1;
}


int Image::getvalue(int i, int j, int k)
{
  return (int)imgdata[(i)*cols*channels + (j)*channels + k];
}


void Image::setvalue(int i, int j, int k, int value)
{
  imgdata[(i)*cols*channels + (j)*channels + (k)] = value;
}




#endif /*__IMAGE_H__*/
