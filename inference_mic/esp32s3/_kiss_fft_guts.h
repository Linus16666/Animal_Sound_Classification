/*
 * Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef KISS_FFT_GUTS_H
#define KISS_FFT_GUTS_H

#include "kiss_fft.h"

#define MAXFACTORS 32

/* e.g. fixed point would use integer types  */
#define KISS_FFT_COS(phase) (kiss_fft_scalar) cosf(phase)
#define KISS_FFT_SIN(phase) (kiss_fft_scalar) sinf(phase)
#define HALF_OF(x) ((x) * .5f)

#define kf_cexp(x,phase) \
    do { \
        (x)->r = KISS_FFT_COS(phase); \
        (x)->i = KISS_FFT_SIN(phase); \
    } while(0)

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

/* a += b */
#define C_ADDTO(res,a) \
    do { \
        (res).r += (a).r; \
        (res).i += (a).i; \
    } while(0)

/* a -= b */
#define C_SUBFROM(res,a) \
    do { \
        (res).r -= (a).r; \
        (res).i -= (a).i; \
    } while(0)

/* m = a * b */
#define C_MUL(m,a,b) \
    do { \
        (m).r = (a).r*(b).r - (a).i*(b).i; \
        (m).i = (a).r*(b).i + (a).i*(b).r; \
    } while(0)

/* m = a * b (with b conjugated for inverse FFT) */
#define C_MULBYSCALAR(c, s) \
    do { \
        (c).r *= (s); \
        (c).i *= (s); \
    } while(0)

/* res = a + b */
#define C_ADD(res,a,b) \
    do { \
        (res).r = (a).r + (b).r; \
        (res).i = (a).i + (b).i; \
    } while(0)

/* res = a - b */
#define C_SUB(res,a,b) \
    do { \
        (res).r = (a).r - (b).r; \
        (res).i = (a).i - (b).i; \
    } while(0)

#define C_FIXDIV(c,div) /* no-op for floating point */
#define S_MUL(a,b) ( (a)*(b) )

struct kiss_fft_state {
    int nfft;
    int inverse;
    int factors[2 * MAXFACTORS];
    kiss_fft_cpx twiddles[1];
};

#endif /* KISS_FFT_GUTS_H */
