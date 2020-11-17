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

Mat SeamCarver::createGradientEnergyMap() {
  int width = im_.cols;
  int height = im_.rows;
  Mat energy = Mat(im_.size(), CV_16U);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      energy.at<ushort>(y, x) = gradientEnergy(im_, y, x);
    }
  }

  Mat scaled;
  convertScaleAbs(energy, scaled);
  return scaled;
}

Mat SeamCarver::createSobelEnergyMap() {
  Mat grey;
  cvtColor(im_, grey, COLOR_BGR2GRAY);

  Mat sobelX, sobelY;
  Sobel(grey, sobelX, CV_16S, 1, 0, 3);
  Sobel(grey, sobelY, CV_16S, 0, 1, 3);

  Mat scaledX, scaledY;
  convertScaleAbs(sobelX, scaledX);
  convertScaleAbs(sobelY, scaledY);

  Mat energy;
  addWeighted(scaledX, 0.5, scaledY, 0.5, 0, energy);
  return energy;
}

Mat SeamCarver::createEnergyMap() {
  if (logging > 0) {
    cout << "Creating energy map" << endl;
  }
  switch (energyFunction_) {
  case Energy::Gradient:
    return createGradientEnergyMap();
  case Energy::Sobel3:
    return createSobelEnergyMap();
  default:
    abort();
  }
}

Mat SeamCarver::createAccumulativeEnergyMap(Mat energy) {
  if (logging > 0) {
    cout << "Creating accumulative energy map" << endl;
  }
  Mat accu = Mat::zeros(im_.size(), CV_32S);
  // copy initial value
  if (dimension_ == Dimension::Vertical) {
    energy.row(0).copyTo(accu.row(0));
  } else {
    energy.col(0).copyTo(accu.col(0));
  }

  unsigned int a, b, c;
  if (dimension_ == Dimension::Vertical) {
    // for each pixel least energetic pixel of the 3 above is added
    for (int y = 1; y < im_.rows; ++y) {
      for (int x = 0; x < im_.cols; ++x) {
        a = accu.at<unsigned int>(y - 1, max(x - 1, 0));
        b = accu.at<unsigned int>(y - 1, x);
        c = accu.at<unsigned int>(y - 1, min(x + 1, im_.cols - 1));
        accu.at<unsigned int>(y, x) = energy.at<uchar>(y, x) + min(a, min(b, c));
      }
    }
  } else {
    for (int x = 1; x < im_.cols; ++x) {
      for (int y = 0; y < im_.rows; ++y) {
        a = accu.at<int>(max(y - 1, 0), x - 1);
        b = accu.at<int>(y, x - 1);
        c = accu.at<int>(min(y + 1, im_.rows - 1), x - 1);
        accu.at<int>(y, x) = energy.at<int>(y, x) + min(a, min(b, c));
      }
    }
  }
  return accu;
}

vector<int> SeamCarver::findOptimalSeam(const Mat &AccuEnergy) {
  if (logging > 0) {
    cout << "Searching Seam" << endl;
  }
  vector<int> seam;
  int a, b, c;
  if (dimension_ == Dimension::Vertical) {
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
      c = AccuEnergy.at<int>(min(current + 1, im_.rows - 1), x + 1);
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
  if (logging > 1) {
    cout << "Chose seam ";
    for (int i : seam) {
      cout << i << ' ';
    }
    cout << endl;
  }
  return seam;
}

void SeamCarver::carveSeam(vector<int> seam) {
  if (logging > 0) {
    cout << "Carving Seam" << endl;
  }
  if (dimension_ == Dimension::Vertical) {
    for (int y = 0; y < im_.rows; ++y) {
      for (int x = seam[y]; x < im_.cols - 1; ++x) {
        im_.at<Vec3b>(y, x) = im_.at<Vec3b>(y, x + 1);
      }
    }
    im_ = im_.colRange(0, im_.cols - 1);
  } else {
    for (int x = 0; x < im_.cols; ++x) {
      for (int y = seam[x]; y < im_.cols - 1; ++y) {
        im_.at<Vec3b>(y, x) = im_.at<Vec3b>(y + 1, x);
      }
    }
    im_ = im_.rowRange(0, im_.rows - 1);
  }
}

void SeamCarver::reduce(int n) {
  for (int i = 0; i < n; ++i) {
    if (logging > 0) {
      cout << format("----- Carving seam #%d -----", i+1) << endl;
    }
    Mat energy = createEnergyMap();
    Mat accuEnergyMap = createAccumulativeEnergyMap(energy);
    vector<int> seam = findOptimalSeam(accuEnergyMap);
    carveSeam(seam);
  }
}

bool SeamCarver::writeImage(const string& path) {
  return imwrite(path, im_);
}

void SeamCarver::showImage() {
  cv::namedWindow("image", WINDOW_AUTOSIZE);
  imshow("image", im_);
  waitKey(0);
}

void SeamCarver::setLogLevel(int level) {
  logging = level;
}
