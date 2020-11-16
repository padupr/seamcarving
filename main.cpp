#include <iostream>
#include <cstring>
#include <getopt.h>
#include <opencv2/opencv.hpp>

#include "SeamCarver.h"

using namespace std;
using namespace cv;

enum class Energy { gradient, dualGradient, sobel };
static constexpr char gradientStr[] = "gradient";
static constexpr char dualGradientStr[] = "dualGradient";
static constexpr char sobelStr[] = "sobel";

void printArgparse(bool vertical, int seams, const Energy &energy);

int main(int argc, char *argv[]) {
  int c;
  int index;

  // options and defaults
  bool logging = false;
  bool vertical = true;
  int seams = -1;
  Energy energy = Energy::gradient;

  while ((c = getopt(argc, argv, "lhvs:e:")) != -1) {
    switch (c) {
    case 'l':
      logging = true;
      break;
    case 'h':
      vertical = false;
      break;
    case 'v':
      vertical = true;
      break;
    case 's':
      seams = std::stoi(optarg);
      if (seams < 0) {
        std::cerr << "-s requires positive integers." << std::endl;
        abort();
      }
      break;
    case 'e':
      if (strcmp(optarg, gradientStr) == 0) {
        energy = Energy::gradient;
      } else if (strcmp(optarg, dualGradientStr) == 0) {
        energy = Energy::dualGradient;
      } else if (strcmp(optarg, sobelStr) == 0) {
        energy = Energy::sobel;
      } else {
        std::cerr << "Unknown energy option " << optarg << ". Try "
                  << std::endl;
        abort();
      }
      break;
    case '?':
      std::cerr << "Unknown option -" << optopt << std::endl;
    default:
      abort();
    }
  }

  if (logging) {
    printArgparse(vertical, seams, energy);
  }

  for (index = optind; index < argc; index++) {
    if (logging) {
      SeamCarver::Dimension dim = vertical ? SeamCarver::Dimension::Vertical : SeamCarver::Dimension::Horizontal;
      SeamCarver::Energy en;
      switch (energy) {
          case Energy::gradient:
              en = SeamCarver::Energy::Gradient;
              break;
          case Energy::dualGradient:
              en = SeamCarver::Energy::DualGradient;
              break;
          case Energy::sobel:
              en = SeamCarver::Energy::Sobel3;
              break;
      }
      char *path = argv[index];
      std::cout << "Processing " << path << std::endl;
      Mat im = imread(path);
      SeamCarver seamCarver(im, dim, en);
      seamCarver.reduce(seams);
      seamCarver.showImage();
      seamCarver.writeImage(format("%s-out-%d.png", path, seams));
    }
  }

  return 0;
}

void printArgparse(bool vertical, int seams, const Energy &energy) {
  cout << "Performing seamcarving with options: " << endl
       << "\tlogging: enabled" << endl
       << "\tdirection: " << (vertical ? "vertical" : "horizontal") << endl
       << "\tseams: " << seams << endl
       << "\tenergy: ";
  if (energy == Energy::gradient) {
    cout << gradientStr;
  } else if (energy == Energy::dualGradient) {
    cout << dualGradientStr;
  }
  cout << std::endl;
}
