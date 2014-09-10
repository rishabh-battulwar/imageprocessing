#ifndef __IMAGE_P__
#define __IMAGE_P__

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>


namespace imageNS
{

  class ImageP
  {
    private:
      //Imagedata
      int cols, rows;
      int channels;
      unsigned char *Imagedata;
      unsigned char ***ImageBW;

    public:

      void read_image(const char *filename, const char *filetype, int width, int height);
      void write_image(const char *filename, const char *filetype, int cols, int rows);
      void color_to_bw(const char *infile, const char *outfile, const char *filetype, int width, int height);
      void image_resize(const char *, int cols, int rows);
      int operator()(int K, int L, int M)
      {
        return (int)Imagedata[ (K*cols+L)*rows + channels];
      }

  };

  void ImageP::read_image(const char *filename, const char *filetype, int width, int height)
  {
    if (!strcmp(filetype, "color"))
    {
      cols = width;
      rows = height;
      channels = 3;

      // Imagedata = new int **[cols];
      // for(int i = 0; i < cols; i++)
      // {
      //   Imagedata[i] = new int *[rows];
      //   for(int j = 0; j < rows; j++)
      //     Imagedata[i][j] = new int[channels];
      // }

      Imagedata = new unsigned char [cols*rows*channels];

      std::cout << "helloin" << std::endl;

      FILE *file;

      if (!(file=fopen(filename,"rb"))) 
      {
        std::cout << "Cannot open file: " << filename << std::endl;
        exit(1);
      }
      if (!(fread(Imagedata, sizeof(int), cols*rows*channels, file)))
        std::cout << "Error Reading" << std::endl;
      fclose(file);  
      //std::cout << "This is the data " << (0,41,0) << " ends here" << std::endl;
      std::cout << "helloo" <<std::endl;
    }

    else if (!strcmp(filetype, "bw"))
    {
      cols = width;
      rows = height;
      channels = 1;
      
      ImageBW = new unsigned char **[cols];
      for(int i = 0; i < cols; i++)
      {
        ImageBW[i] = new unsigned char *[rows];
        for(int j = 0; j < rows; j++)
          ImageBW[i][j] = new unsigned char[channels];
      }

      FILE *file;

      if (!(file=fopen(filename,"rb"))) 
      {
        std::cout << "Cannot open file: " << filename << std::endl;
        exit(1);
      }
      if (!(fread(ImageBW, sizeof(unsigned char), cols*rows*channels, file)))
        std::cout << "Error Reading" << std::endl;
      fclose(file);  

    }

  }

  void ImageP::write_image(const char *filename, const char *filetype, int width, int height)
  {
    if (!strcmp(filetype, "color"))
    {
      cols = width;
      rows = height;
      channels = 3;

      FILE *file;
      if (!(file=fopen(filename,"wb"))) 
      {
        std::cout << "Cannot open file: " << filename << std::endl;
        exit(1);
      }
     // if (!(fwrite(Imagedata, sizeof(unsigned char), cols*rows*channels, file)))
         std::cout << "Error Writing" << std::endl;
      fclose(file);
      
      // for(int i = 0; i < size; i++)
      // {
      //   for(int j = 0; j < size; j++)
      //   {
      //     delete [] Imagedata[i][j];
      //   }
      //   delete [] Imagedata[i];
      // }
      // delete [] Imagedata;

    }

    else if (!strcmp(filetype, "bw"))
    {
      cols = width;
      rows = height;
      channels = 1;

      FILE *file;

      if (!(file=fopen(filename,"wb"))) 
      {
        std::cout << "Cannot open file: " << filename << std::endl;
        exit(1);
      }
      if (!(fwrite(ImageBW, sizeof(unsigned char), cols*rows*channels, file)))
        std::cout << "Error Writing" << std::endl;
      fclose(file);  

    }

  }

  void ImageP::color_to_bw(const char *infile, const char *outfile, const char *filetype, int width, int height)
  {
    std::cout << "hello1";
    read_image(infile, filetype, width, height);
    // std::cout<< "helloout" << std::endl;
    // filetype = "bw";
    // ImageBW = new unsigned char **[width];
    // for(int i = 0; i < width; i++)
    // {
    //   ImageBW[i] = new unsigned char *[height];
    //   for(int j = 0; j < height; j++)
    //     ImageBW[i][j] = new unsigned char;
    // }

    // for(int i = 0; i < width; i++)
    //   for(int j = 0; j < height; j++)
    //     //ImageBW[i][j] = Imagedata[i][j][0] * 0.21+Imagedata[i][j][1] * 0.72+Imagedata[i][j][2] * 0.07;
    // //std::cout << Imagedata[34][5][2];
    // write_image(outfile, filetype, width, height);
  }

};

#endif /*__IMAGE_IO__*/
