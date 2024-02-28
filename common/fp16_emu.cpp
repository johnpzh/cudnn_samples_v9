#include "fp16_emu.h"

#define STATIC_ASSERT(cond)                                \
    do {                                                   \
        typedef char compile_time_assert[(cond) ? 1 : -1]; \
    } while (0)

// Host functions for converting between FP32 and FP16 formats
// Paulius Micikevicius (pauliusm@nvidia.com)

half1
cpu_float2half_rn(float f) {
    unsigned x = *((int *)(void *)(&f));
    unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    unsigned sign, exponent, mantissa;

    __half_raw hr;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
        hr.x = 0x7fffU;
        return reinterpret_cast<half1 &>(hr);
    }

    sign = ((x >> 16) & 0x8000);

    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
        hr.x = sign | 0x7c00U;
        return reinterpret_cast<half1 &>(hr);
    }
    if (u < 0x33000001) {
        hr.x = sign | 0x0000U;
        return reinterpret_cast<half1 &>(hr);
    }

    exponent = ((u >> 23) & 0xff);
    mantissa = (u & 0x7fffff);

    if (exponent > 0x70) {
        shift = 13;
        exponent -= 0x70;
    } else {
        shift    = 0x7e - exponent;
        exponent = 0;
        mantissa |= 0x800000;
    }
    lsb    = (1 << shift);
    lsb_s1 = (lsb >> 1);
    lsb_m1 = (lsb - 1);

    // Round to nearest even.
    remainder = (mantissa & lsb_m1);
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
        ++mantissa;
        if (!(mantissa & 0x3ff)) {
            ++exponent;
            mantissa = 0;
        }
    }

    hr.x = (sign | (exponent << 10) | mantissa);

    return reinterpret_cast<half1 &>(hr);
}

float
cpu_half2float(half1 h) {
    STATIC_ASSERT(sizeof(int) == sizeof(float));

    __half_raw hr = reinterpret_cast<__half_raw &>(h);

    unsigned sign     = ((hr.x >> 15) & 1);
    unsigned exponent = ((hr.x >> 10) & 0x1f);
    unsigned mantissa = ((hr.x & 0x3ff) << 13);

    if (exponent == 0x1f) { /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) { /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1; /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff; /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }

    int temp = ((sign << 31) | (exponent << 23) | mantissa);

    return reinterpret_cast<float &>(temp);
}

float
convertToFloat(void *x, cudnnDataType_t dataType) {
    if (dataType == CUDNN_DATA_HALF) {
        half1 h = *static_cast<half1 *>(x);
        return cpu_half2float(h);
    } else if (dataType == CUDNN_DATA_INT8x4) {
        int8_t tmp = *static_cast<int8_t *>(x);
        return float(tmp);
    } else if (dataType == CUDNN_DATA_INT8x32) {
        int32_t tmp = *static_cast<int32_t *>(x);
        return float(tmp);
    } else if (dataType == CUDNN_DATA_FLOAT) {
        float f = *static_cast<float *>(x);
        return f;
    }
}

void
convertFromFloat(float *x, cudnnDataType_t dataType, void *result) {
    if (x == NULL) {
        return;
    }

    float tmp = *static_cast<float *>(x);
    if (dataType == CUDNN_DATA_HALF) {
        *(half1 *)result = (cpu_float2half_rn(tmp));
    } else if (dataType == CUDNN_DATA_INT8x4) {
        *(int8_t *)result = (int8_t)tmp;
    } else if (dataType == CUDNN_DATA_INT8x32) {
        *(int32_t *)result = (int32_t)tmp;
    } else if (dataType == CUDNN_DATA_FLOAT) {
        *(float *)result = tmp;
    }
}
