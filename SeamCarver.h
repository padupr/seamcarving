#ifndef SEAMCARVING_SEAMCARVING_H
#define SEAMCARVING_SEAMCARVING_H

#include <opencv2/opencv.hpp>
#include <utility>

using namespace std;
using namespace cv;

class SeamCarver {
public:
  enum DIMENSION { HORIZONTAL, VERTICAL };
  enum ENERGY { GRADIENT, DUALGRADIENT };

  SeamCarver(Mat im, DIMENSION dim, ENERGY e)
      : im{std::move(im)}, dimension{dim}, energyFunction{e} {}
  explicit SeamCarver(Mat im) : im{std::move(im)} {}

  void reduce(int n);
  void writeImage(const string& path);
  void showImage();

private:
  cv::Mat im;
  DIMENSION dimension = VERTICAL;
  ENERGY energyFunction = GRADIENT;
  Mat createEnergyMap();
  Mat createAccumulativeEnergyMap(Mat energy);
  vector<int> findOptimalSeam(const Mat& accuEnergy);
  void carveSeam(vector<int> seam);
  Mat createGradientEnergyMap();
  Mat createDualGradientEnergyMap();
};

#endif // SEAMCARVING_SEAMCARVING_H
