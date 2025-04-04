# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")

# Project name
project(LeidenCommunitiesOpenMP VERSION 1.0)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Limited configurations" FORCE)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FindOpenMP)

# Enable OpenMP support (needed for parallelization)
find_package(OpenMP REQUIRED)

# Add the main source file
add_executable(leiden main.cxx)

