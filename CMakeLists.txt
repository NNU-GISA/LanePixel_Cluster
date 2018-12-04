cmake_minimum_required(VERSION 3.12)
project(hough_transform)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hough_transform main.cpp)
target_link_libraries(hough_transform ${OpenCV_LIBS})