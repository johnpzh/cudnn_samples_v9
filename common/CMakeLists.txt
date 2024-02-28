add_library(
    fp16_dev OBJECT
    fp16_dev.cu
)

add_library(
    fp16_emu OBJECT
    fp16_emu.cpp
)

target_link_libraries(
    fp16_dev
    fp16_emu
    CUDA::cudart_static
)

target_link_libraries(
    fp16_emu
    CUDA::cudart_static
    cudnn
)

target_include_directories(fp16_dev PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(fp16_emu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
