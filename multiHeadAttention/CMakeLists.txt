add_executable(
    multiHeadAttention
    multiHeadAttention.cpp
)

target_link_libraries(
    multiHeadAttention
    CUDA::cublas
    CUDA::cudart_static
    cudnn
    fp16_emu
)

list(APPEND datfiles
    "${CMAKE_CURRENT_BINARY_DIR}/dk.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/dout.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/dq.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/dv.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/dwk.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/dwo.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/dwq.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/dwv.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/k.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/meta.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/out.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/q.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/v.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/wk.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/wo.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/wq.dat"
    "${CMAKE_CURRENT_BINARY_DIR}/wv.dat"
)

add_test(
    NAME setupMultiHeadAttentionTest
    COMMAND multiHeadAttention
        "-attnFileDump1"
        "-attnTrain1"
        "-attnDataType0"
        "-attnCompPrec0"
        "-attnResLink1"
        "-attnDataLayout3"
        "-attnNumHeads3"
        "-attnBeamSize1"
        "-attnBatchSize1"
        "-attnQsize8"
        "-attnKsize8"
        "-attnVsize8"
        "-attnProjQsize2"
        "-attnProjKsize2"
        "-attnProjVsize2"
        "-attnProjOsize8"
        "-attnResLink0"
        "-attnProjBias0"
        "-attnSeqLenQ4"
        "-attnSeqLenK10"
        "-attnSmScaler1.0"
        "-attnRandSeed1234"
)

add_test(
    NAME multiHeadAttentionTest
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/attn_ref.py
)

set_tests_properties(setupMultiHeadAttentionTest PROPERTIES FIXTURES_SETUP MHA)
set_tests_properties(multiHeadAttentionTest PROPERTIES FIXTURES_REQUIRED MHA)
