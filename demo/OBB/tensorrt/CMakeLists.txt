cmake_minimum_required(VERSION 3.8)

project(yolox_obb)
add_definitions(-std=c++14)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
# IF(POLICY CMP0012)
#     CMAKE_POLICY(SET CMP0012 NEW)
# ENDIF()
# CMAKE_POLICY(SET CMP0074 NEW)

# 本目录
include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

find_package(CUDA REQUIRED)

include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)


include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGES} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${CMAKE_PROJECT_NAME} ${DIR_SRCS})
target_link_libraries(${CMAKE_PROJECT_NAME} nvinfer nvinfer_plugin)
target_link_libraries(${CMAKE_PROJECT_NAME} cudart)

target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})

add_definitions(-O0 -pthread)
