cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(tsdf_fusion)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3 -fopenmp -fPIC -Wno-deprecated -Wenum-compare -std=c++11")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# OpenCV
find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# PCL
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# VTK
get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
    if(it MATCHES "vtk*")
        list(APPEND vtk_flags ${it})
    endif()
endforeach()

foreach(d ${vtk_flags})
    remove_definitions(-D${d})
endforeach()

# CUDA
find_package(CUDA REQUIRED)
#set(CUDA_NVCC_FLAGS -O3;-G;-g)

# 使CUDA NVCC 编译器支持C++11特性
list(APPEND CUDA_NVCC_FLAGS -std=c++11;-O3)
list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)

#file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
#file(GLOB CURRENT_SOURCES  *.cpp *.cu)

cuda_add_executable (tsdf_fusion main.cpp fusion.cu)

target_link_libraries(tsdf_fusion ${OpenCV_LIBS} ${PCL_LIBRARIES})