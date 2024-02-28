#if !defined(_FP16_DEV_H_)
#define _FP16_DEV_H_

#include "fp16_emu.h"

template <class value_type>
void
gpu_float2half_rn(int size, const value_type *buffIn, half1 *buffOut);

#endif  // _FP16_DEV_H_
