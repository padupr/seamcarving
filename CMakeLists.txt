cmake_minimum_required(VERSION 3.17)
project(seamcarving)

set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(seamcarving main.cpp SeamCarver.cpp)

target_link_libraries( seamcarving ${OpenCV_LIBS} )