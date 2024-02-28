add_executable(
    conv_sample
        conv_sample.cpp
)

target_link_libraries(
    conv_sample
        CUDA::cublas
        CUDA::cudart_static
        cudnn
        fp16_dev
        fp16_emu
)

#Use the following arguments to run sample with different convolution parameters:
add_test(
    NAME conv_sample_test_0
    COMMAND conv_sample
        "-c2048" "-h7" "-w7" "-k512" "-r1" "-s1" "-pad_h0" "-pad_w0" "-u1" "-v1"
)

add_test(
    NAME conv_sample_test_1
    COMMAND conv_sample
        "-c512" "-h28" "-w28" "-k128" "-r1" "-s1" "-pad_h0" "-pad_w0" "-u1" "-v1"
)

add_test(
    NAME conv_sample_test_2
    COMMAND conv_sample
        "-c512" "-h28" "-w28" "-k1024" "-r1" "-s1" "-pad_h0" "-pad_w0" "-u2" "-v2"
)

add_test(
    NAME conv_sample_test_3
    COMMAND conv_sample
        "-c512" "-h28" "-w28" "-k256" "-r1" "-s1" "-pad_h0" "-pad_w0" "-u2" "-v2"
)

add_test(
    NAME conv_sample_test_4
    COMMAND conv_sample
        "-c256" "-h14" "-w14" "-k256" "-r3" "-s3" "-pad_h1" "-pad_w1" "-u1" "-v1"
)

add_test(
    NAME conv_sample_test_5
    COMMAND conv_sample
        "-c256" "-h14" "-w14" "-k1024" "-r1" "-s1" "-pad_h0" "-pad_w0" "-u1" "-v1"
)

add_test(
    NAME conv_sample_test_6
    COMMAND conv_sample
        "-c1024" "-h14" "-w14" "-k256" "-r1" "-s1" "-pad_h0" "-pad_w0" "-u1" "-v1"
)

add_test(
    NAME conv_sample_test_7
    COMMAND conv_sample
        "-c1024" "-h14" "-w14" "-k2048" "-r1" "-s1" "-pad_h0" "-pad_w0" "-u2" "-v2"
)

add_test(
    NAME conv_sample_test_8
    COMMAND conv_sample
        "-c1024" "-h14" "-w14" "-k512" "-r1" "-s1" "-pad_h0" "-pad_w0" "-u2" "-v2"
)

add_test(
    NAME conv_sample_test_9
    COMMAND conv_sample
        "-c512" "-h7" "-w7" "-k512" "-r3" "-s3" "-pad_h1" "-pad_w1" "-u1" "-v1"
)

add_test(
    NAME conv_sample_test_10
    COMMAND conv_sample
        "-c512" "-h7" "-w7" "-k2048" "-r1" "-s1" "-pad_h0" "-pad_w0" "-u1" "-v1"
)

add_test(
    NAME conv_sample_test_11
    COMMAND conv_sample
        "-c2048" "-h7" "-w7" "-k512" "-r1" "-s1" "-pad_h0" "-pad_w0" "-u1" "-v1"
)

#Use the following arguments to run sample with int8x4 and int8x32 benchmarks:
add_test(
    NAME conv_sample_test_12
    COMMAND conv_sample
        "-mathType1" "-filterFormat2" "-dataType2" "-n1" "-c512" "-h100" "-w100" "-k64" "-r8" "-s8" "-pad_h0" "-pad_w0" "-u1" "-v1" "-b"
)

add_test(
    NAME conv_sample_test_13
    COMMAND conv_sample
        "-mathType1" "-filterFormat2" "-dataType2" "-n1" "-c4096" "-h64" "-w64" "-k64" "-r4" "-s4" "-pad_h1" "-pad_w1" "-u1" "-v1" "-b"
)

add_test(
    NAME conv_sample_test_14
    COMMAND conv_sample
        "-mathType1" "-filterFormat2" "-dataType2" "-n1" "-c512" "-h100" "-w100" "-k64" "-r8" "-s8" "-pad_h1" "-pad_w1" "-u1" "-v1" "-b"
)

add_test(
    NAME conv_sample_test_15
    COMMAND conv_sample
        "-mathType1" "-filterFormat2" "-dataType2" "-n1" "-c512" "-h128" "-w128" "-k64" "-r13" "-s13" "-pad_h1" "-pad_w1" "-u1" "-v1" "-b"
)

add_test(
    NAME conv_sample_test_16
    COMMAND conv_sample
        "-mathType1" "-filterFormat2" "-dataType3" "-n1" "-c512" "-h100" "-w100" "-k64" "-r8" "-s8" "-pad_h0" "-pad_w0" "-u1" "-v1" "-b"
)

add_test(
    NAME conv_sample_test_17
    COMMAND conv_sample
        "-mathType1" "-filterFormat2" "-dataType3" "-n1" "-c4096" "-h64" "-w64" "-k64" "-r4" "-s4" "-pad_h1" "-pad_w1" "-u1" "-v1" "-b"
)

add_test(
    NAME conv_sample_test_18
    COMMAND conv_sample
        "-mathType1" "-filterFormat2" "-dataType3" "-n1" "-c512" "-h100" "-w100" "-k64" "-r8" "-s8" "-pad_h1" "-pad_w1" "-u1" "-v1" "-b"
)

add_test(
    NAME conv_sample_test_19
    COMMAND conv_sample
        "-mathType1" "-filterFormat2" "-dataType3" "-n1" "-c512" "-h128" "-w128" "-k64" "-r13" "-s13" "-pad_h1" "-pad_w1" "-u1" "-v1" "-b"
)

add_test(
    NAME conv_sample_test_20
    COMMAND conv_sample
        "-mathType1" "-filterFormat2" "-dataType3" "-n5" "-c32" "-h16" "-w16" "-k32" "-r5" "-s5" "-pad_h0" "-pad_w0" "-u1" "-v1" "-b" "-transformFromNCHW"
)

#Use the following arguments to run sample dgrad with folding:
add_test(
    NAME conv_sample_test_21
    COMMAND conv_sample
        "-dgrad" "-c1024" "-h14" "-w14" "-k2048" "-r1" "-s1" "-pad_h0" "-pad_w0" "-u2" "-v2" "-fold"
)