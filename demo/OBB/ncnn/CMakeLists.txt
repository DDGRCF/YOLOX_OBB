cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

project(yolox_ncnn_demo)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
if (OPENCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${OpenCV_LIBS})
endif()

if (NOT DEFINED NCNN_DIR) 
    message(FATAL_ERROR "Can't find NCNN_DIR Var")
else()
    set(ncnn_DIR ${NCNN_DIR} CACHE PATH "Directory that contains ncnnConfig.cmake")
    find_package(ncnn REQUIRED)
    if (ncnn_FOUND) 
        set(EXTRA_LIBS ${EXTRA_LIBS} ncnn)
    endif()
endif()

include_directories(${NCNN_INCLUDE_DIRS})

add_executable(${CMAKE_PROJECT_NAME} main.cc)

target_link_libraries(${CMAKE_PROJECT_NAME} ${EXTRA_LIBS})