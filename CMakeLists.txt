cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 11)

if(NOT DEFINED PROJECT_NAME)
    project(cudnn-samples LANGUAGES CUDA CXX)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
endif()

find_package(CUDAToolkit REQUIRED)
if(NOT TARGET cudnn)
    find_package(cuDNN REQUIRED)
else()
    include_directories(${CMAKE_SOURCE_DIR}/include)
endif()

enable_testing()

add_subdirectory(common)
add_subdirectory(conv_sample)
add_subdirectory(mnistCUDNN)
add_subdirectory(multiHeadAttention)
add_subdirectory(RNN_v8.0)
