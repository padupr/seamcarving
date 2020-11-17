#include <cstring>
#include <getopt.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "SeamCarver.h"

using namespace std;
using namespace cv;

enum class Energy { gradient, sobel };
static constexpr char gradientStr[] = "gradient";
static constexpr char sobelStr[] = "sobel";

void printArgparse(bool vertical, int seams, const Energy &energy);
void printUsage();

SeamCarver::Energy convertEnergy(const Energy &energy);
int main(int argc, char *argv[]) {
    // options and defaults
    int logging = 0;
    bool vertical = true;
    int seams = -1;
    Energy energy = Energy::gradient;

    opterr = 0;
    int c;
    while ((c = getopt(argc, argv, "l:hvs:e:")) != -1) {
        switch (c) {
        case 'l':
            logging = std::stoi(optarg);
            if (logging < 0 || logging >= 3) {
                std::cerr << "Logging (-l) must be set to value 0, 1, or 2."
                          << std::endl;
                return 1;
            }
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
                return 1;
            }
            break;
        case 'e':
            if (strcmp(optarg, gradientStr) == 0) {
                energy = Energy::gradient;
            } else if (strcmp(optarg, sobelStr) == 0) {
                energy = Energy::sobel;
            } else {
                std::cerr << "Unknown energy option " << optarg << ". Try "
                          << std::endl;
                return 1;
            }
            break;
        case '?':
            std::cout << "Unknown option -" << (char)optopt << std::endl;
            printUsage();
        default:
            return 1;
        }
    }

    if (seams <= 0) {
        std::cerr << "The number of seams needs to be larger than 0"
                  << std::endl;
        return 1;
    }

    if (logging > 1) {
        printArgparse(vertical, seams, energy);
    }

    for (int index = optind; index < argc; index++) {
        SeamCarver::Dimension dim = vertical
                                        ? SeamCarver::Dimension::Vertical
                                        : SeamCarver::Dimension::Horizontal;
        SeamCarver::Energy en = convertEnergy(energy);
        char *path = argv[index];

        if (logging > 0) {
            std::cout << "Processing " << path << std::endl;
        }

        Mat im = imread(path);
        if (vertical && im.rows <= seams) {
            std::cerr << "Seams must be less than image width." << std::endl;
            return 1;
        }
        if (!vertical && im.cols <= seams) {
            std::cerr << "Seams must be less than image height." << std::endl;
            return 1;
        }

        SeamCarver seamCarver(im, dim, en);
        seamCarver.setLogLevel(logging);
        seamCarver.reduce(seams);
        seamCarver.showImage();
        string outPath = format("%s-out-%d.png", path, seams);
        if (seamCarver.writeImage(outPath) && logging > 0) {
            std::cout << "Written to" << outPath << std::endl;
        } else if (logging > 0) {
            std::cerr << "Could not write to" << outPath << std::endl;
        }
    }

    return 0;
}

SeamCarver::Energy convertEnergy(const Energy &energy) {
    SeamCarver::Energy en;
    switch (energy) {
    case Energy::gradient:
        en = SeamCarver::Gradient;
        break;
    case Energy::sobel:
        en = SeamCarver::Sobel3;
        break;
    }
    return en;
}

void printArgparse(bool vertical, int seams, const Energy &energy) {
    cout << "Performing seamcarving with options: " << endl
         << "\tlogging: enabled" << endl
         << "\tdirection: " << (vertical ? "vertical" : "horizontal") << endl
         << "\tseams: " << seams << endl
         << "\tenergy: ";
    if (energy == Energy::gradient) {
        cout << gradientStr;
    } else if (energy == Energy::sobel) {
        cout << sobelStr;
    }
    cout << std::endl;
}

void printUsage() {
    std::cout
        << "Usage: seamcarving [OPTION]... [FILE]..." << std::endl
        << "Options:" << std::endl
        << "  -l level"
        << "      Logging level. 0 off, 1 info, 2 verbose." << std::endl
        << "  -h"
        << "            Reduce horizontal." << std::endl
        << "  -v"
        << "            Reduce vertical." << std::endl
        << "  -s seams"
        << "      Number of seams to remove." << std::endl
        << "  -e algorithm"
        << "  Select energy calculation from gradient, dualGradient, and sobel."
        << std::endl
        << std::endl
        << "This tool implements seam carving for content-aware image "
           "downsizing."
        << std::endl;
}
