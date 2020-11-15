#include "SeamCarver.h"
#include <opencv2/opencv.hpp>

using namespace cv;

inline int calculatePixelDistance(const Vec3b &first, const Vec3b &second) {
  int blue = (first.val[0] - second.val[0]);
  blue *= blue;
  int green = (first.val[1] - second.val[1]);
  green *= green;
  int red = (first.val[2] - second.val[2]);
  red *= red;
  return sqrt(blue + green + red);
}

inline int gradientEnergy(const Mat &im, int y, int x) {
  const auto &pixel = im.at<Vec3b>(y, x);
  Vec3b other;

  int energy = 0;
  other = im.at<Vec3b>(y, max(x - 1, 0));
  energy += calculatePixelDistance(pixel, other) / 2;
  other = im.at<Vec3b>(max(y - 1, 0), x);
  energy += calculatePixelDistance(pixel, other) / 2;
  return energy;
}

inline int dualGradientEnergy(const Mat &im, int y, int x) {
  const auto &pixel = im.at<Vec3b>(y, x);
  Vec3b other;

  int energy = 0;
  other = im.at<Vec3b>(y, max(x - 1, 0));
  energy += calculatePixelDistance(pixel, other) / 2;
  other = im.at<Vec3b>(max(y - 1, 0), x);
  energy += calculatePixelDistance(pixel, other) / 2;
  other = im.at<Vec3b>(y, min(x + 1, im.cols - 1));
  energy += calculatePixelDistance(pixel, other) / 2;
  other = im.at<Vec3b>(min(y + 1, im.rows - 1), x);
  energy += calculatePixelDistance(pixel, other) / 2;
  return energy;
}

Mat SeamCarver::createGradientEnergyMap() {
  int width = im.cols;
  int height = im.rows;
  Mat energy = Mat(im.size(), CV_16U);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      energy.at<ushort>(y, x) = gradientEnergy(im, y, x);
    }
  }

  Mat scaled;
  convertScaleAbs(energy, scaled);
  return scaled;
}

Mat SeamCarver::createDualGradientEnergyMap() {
  int width = im.cols;
  int height = im.rows;
  Mat energy = Mat(im.size(), CV_16U);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      energy.at<ushort>(y, x) = dualGradientEnergy(im, y, x);
    }
  }

  Mat scaled;
  convertScaleAbs(energy, scaled);
  return scaled;
}

Mat SeamCarver::createEnergyMap() {
  switch (energyFunction) {
  case ENERGY::GRADIENT:
    return createGradientEnergyMap();
  case ENERGY::DUALGRADIENT:
    return createDualGradientEnergyMap();
  default:
    abort();
  }
}

Mat SeamCarver::createAccumulativeEnergyMap(Mat energy) {
  Mat cumul = Mat::zeros(im.size(), CV_32S);
  // copy initial value
  if (dimension == DIMENSION::VERTICAL) {
    energy.row(0).copyTo(cumul.row(0));
  } else {
    energy.col(0).copyTo(cumul.col(0));
  }

  unsigned int a, b, c;
  if (dimension == DIMENSION::VERTICAL) {
    // for each pixel least energetic pixel of the 3 above is added
    for (int y = 1; y < im.rows; ++y) {
      for (int x = 0; x < im.cols; ++x) {
        a = cumul.at<unsigned int>(y - 1, max(x - 1, 0));
        b = cumul.at<unsigned int>(y - 1, x);
        c = cumul.at<unsigned int>(y - 1, min(x + 1, im.cols - 1));
        cumul.at<unsigned int>(y, x) = energy.at<uchar>(y, x) + min(a, min(b, c));
      }
    }
  } else {
    for (int x = 1; x < im.cols; ++x) {
      for (int y = 0; y < im.rows; ++y) {
        a = cumul.at<int>(max(y - 1, 0), x - 1);
        b = cumul.at<int>(y, x - 1);
        c = cumul.at<int>(min(y + 1, im.rows - 1), x - 1);
        cumul.at<int>(y, x) = energy.at<int>(y, x) + min(a, min(b, c));
      }
    }
  }
  return cumul;
}

vector<int> SeamCarver::findOptimalSeam(const Mat &AccuEnergy) {
  vector<int> seam;
  int a, b, c;
  if (dimension == DIMENSION::VERTICAL) {
    int rows = AccuEnergy.rows;
    seam = vector<int>(rows);

    Mat last_row = AccuEnergy.row(rows - 1);
    int current = min_element(last_row.begin<int>(), last_row.end<int>()) -
                  last_row.begin<int>();
    seam[rows - 1] = current;

    for (int y = rows - 2; y >= 0; --y) {
      a = AccuEnergy.at<int>(y + 1, max(current - 1, 0));
      b = AccuEnergy.at<int>(y + 1, current);
      c = AccuEnergy.at<int>(y + 1, min(current + 1, AccuEnergy.cols - 1));
      if (a < b && a < c) {
        current -= 1;
      } else if (b < a && b < c) {
        current = current;
      } else {
        current += 1;
      }
      current = max(min(current, AccuEnergy.cols - 1), 0);
      seam[y] = current;
    }
  } else {
    int cols = AccuEnergy.cols;
    seam = vector<int>(cols);

    Mat last_col = AccuEnergy.col(cols - 1);
    int current = min_element(last_col.begin<int>(), last_col.end<int>()) -
                  last_col.begin<int>();
    seam[cols - 1] = current;

    for (int x = cols - 2; x >= 0; --x) {
      a = AccuEnergy.at<int>(max(current - 1, 0), x + 1);
      b = AccuEnergy.at<int>(current, x + 1);
      c = AccuEnergy.at<int>(min(current + 1, im.rows - 1), x + 1);
      if (a < b && a < c) {
        current -= 1;
      } else if (b < a && b < c) {
        current = current;
      } else {
        current += 1;
      }
      current = max(min(current, AccuEnergy.rows - 1), 0);
      seam[x] = current;
    }
  }
  return seam;
}

void SeamCarver::carveSeam(vector<int> seam) {
  if (dimension == DIMENSION::VERTICAL) {
    for (int y = 0; y < im.rows; ++y) {
      for (int x = seam[y]; x < im.cols - 1; ++x) {
        im.at<Vec3b>(y, x) = im.at<Vec3b>(y, x + 1);
      }
    }
    im = im.colRange(0, im.cols - 1);
  } else {
    for (int x = 0; x < im.cols; ++x) {
      for (int y = seam[x]; y < im.cols - 1; ++y) {
        im.at<Vec3b>(y, x) = im.at<Vec3b>(y + 1, x);
      }
    }
    im = im.rowRange(0, im.rows - 1);
  }
}

void SeamCarver::reduce(int n) {
  for (int i = 0; i < n; ++i) {
    Mat energy = createEnergyMap();
    Mat accuEnergyMap = createAccumulativeEnergyMap(energy);
    vector<int> seam = findOptimalSeam(accuEnergyMap);
    carveSeam(seam);
  }
}

void SeamCarver::writeImage(const string& path) {
  imwrite(path, im);
}

void SeamCarver::showImage() {
  cv::namedWindow("image", WINDOW_AUTOSIZE);
  imshow("image", im);
  waitKey(0);
}
