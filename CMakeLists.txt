cmake_minimum_required(VERSION 3.12)
project(hough_transform)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(hough_transform main.cpp Solver.h)
target_link_libraries(hough_transform ${OpenCV_LIBS} ${CERES_LIBRARIES})