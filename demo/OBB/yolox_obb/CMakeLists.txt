cmake_minimum_required(VERSION 2.6)

project(yolox_obb)
add_definitions(-std=c++14)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
IF(POLICY CMP0012)
    CMAKE_POLICY(SET CMP0012 NEW)
ENDIF()

# 本目录
include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

find_package(CUDA REQUIRED)

include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)


include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGES} \
    -std=c++14 -Wall -ofast -Wfatal-errors \
    -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolox_obb ${DIR_SRCS})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
target_link_libraries(yolox_obb nvinfer nvinfer_plugin)
target_link_libraries(yolox_obb cudart)

target_link_libraries(yolox_obb ${OpenCV_LIBS})

add_definitions(-O0 -pthread)
