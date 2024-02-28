add_subdirectory(FreeImage)

add_executable(
    mnistCUDNN
    mnistCUDNN.cpp
)

add_test(
    NAME mnistCUDNN_test
    COMMAND mnistCUDNN
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    mnistCUDNN
    PRIVATE FreeImage
    CUDA::cublas
    CUDA::cudart_static
    cudnn
    fp16_dev
    fp16_emu
)

if(WIN32)
    set_property(TARGET mnistCUDNN PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
else()
    target_compile_options(mnistCUDNN PRIVATE -fpermissive)
endif()
