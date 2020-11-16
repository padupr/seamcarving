#ifndef SEAMCARVING_SEAMCARVING_H
#define SEAMCARVING_SEAMCARVING_H

#include <opencv2/opencv.hpp>
#include <utility>

using namespace std;
using namespace cv;

class SeamCarver {
public:
  enum Dimension { Horizontal, Vertical };
  enum Energy { Gradient, DualGradient, Sobel3 };

  SeamCarver(Mat im, Dimension dim, Energy e)
      : im_{std::move(im)}, dimension_{dim}, energyFunction_{e} {}
  explicit SeamCarver(Mat im) : im_{std::move(im)} {}

  void reduce(int n);
  void writeImage(const string& path);
  void showImage();

private:
  cv::Mat im_;
  Dimension dimension_ = Vertical;
  Energy energyFunction_ = Gradient;
  Mat createEnergyMap();
  Mat createAccumulativeEnergyMap(Mat energy);
  vector<int> findOptimalSeam(const Mat& accuEnergy);
  void carveSeam(vector<int> seam);
  Mat createGradientEnergyMap();
  Mat createDualGradientEnergyMap();
  Mat createSobelEnergyMap();
};

#endif // SEAMCARVING_SEAMCARVING_H
