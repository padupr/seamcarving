cmake_minimum_required(VERSION 3.17)
project(seamcarving)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include/opencv4)

add_executable(seamcarving Main.cpp)