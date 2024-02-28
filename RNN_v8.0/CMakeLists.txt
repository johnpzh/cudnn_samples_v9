add_executable(
    RNN_v8.0
    RNN_example.cu
)

target_link_libraries(
    RNN_v8.0
    CUDA::cublas
    CUDA::cudart_static
    cudnn
    fp16_emu
)

foreach(TESTNUM RANGE 1 4)
    math(EXPR MODENUM "${TESTNUM} - 1")
    cmake_path(SET "wd" "${CMAKE_CURRENT_BINARY_DIR}/test_${TESTNUM}")
    cmake_path(SET "result" "${wd}/result.txt")
    add_test(
        NAME "create_dir_RNN8_test_${TESTNUM}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${wd}"
    )
    add_test(
        NAME "setup_RNN8_test_${TESTNUM}"
        COMMAND RNN_v8.0 "-dataType1" "-seqLength20" "-numLayers2" "-inputSize512" "-hiddenSize512" "-projSize512" "-miniBatch64" "-inputMode1" "-dirMode0" "-cellMode${MODENUM}" "-biasMode3" "-algorithm0" "-mathPrecision1" "-mathType0" "-dropout0.0" "-printWeights0"
        WORKING_DIRECTORY "${wd}"
    )
    add_test(
        NAME "RNN8_test_${TESTNUM}"
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/compare.py "${result}" "${CMAKE_CURRENT_SOURCE_DIR}/golden_${TESTNUM}.txt"
    )
    set_tests_properties("setup_RNN8_test_${TESTNUM}" PROPERTIES DEPENDS "create_dir_RNN8_test_${TESTNUM}")
    set_tests_properties("create_dir_RNN8_test_${TESTNUM}" PROPERTIES FIXTURES_SETUP "fixture_${TESTNUM}")
    set_tests_properties("setup_RNN8_test_${TESTNUM}" PROPERTIES FIXTURES_SETUP "fixture_${TESTNUM}")
    set_tests_properties("RNN8_test_${TESTNUM}" PROPERTIES FIXTURES_REQUIRED "fixture_${TESTNUM}")
endforeach()