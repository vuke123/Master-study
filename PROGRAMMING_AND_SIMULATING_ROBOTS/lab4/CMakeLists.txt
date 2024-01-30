set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.10)
project(proba)

# Find Eigen library
find_package(Eigen3 REQUIRED)

# Set up executable
add_executable(proba luka_vukelic.cpp)

# Include Eigen headers
target_include_directories(proba PRIVATE ${EIGEN3_INCLUDE_DIR})
