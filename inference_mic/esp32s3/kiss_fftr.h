/*
 * Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef KISS_FFTR_H
#define KISS_FFTR_H

#include "kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Real-optimized version can save about 45% cpu time vs. complex transform
 * when doing an FFT of purely real input.
 */

typedef struct kiss_fftr_state * kiss_fftr_cfg;

/*
 *  nfft must be even
 *
 *  If you don't care to allocate memory yourself,
 *  kiss_fftr_alloc(nfft, 0, NULL, NULL) will allocate for you.
 */
kiss_fftr_cfg kiss_fftr_alloc(int nfft, int inverse_fft,
                               void * mem, size_t * lenmem);

/*
 * input timedata has nfft scalar points
 * output freqdata has nfft/2+1 complex points
 */
void kiss_fftr(kiss_fftr_cfg cfg, const kiss_fft_scalar * timedata,
               kiss_fft_cpx * freqdata);

/*
 * input freqdata has nfft/2+1 complex points
 * output timedata has nfft scalar points
 * NOTE: the output is NOT scaled by 1/nfft — divide by nfft yourself.
 */
void kiss_fftri(kiss_fftr_cfg cfg, const kiss_fft_cpx * freqdata,
                kiss_fft_scalar * timedata);

#define kiss_fftr_free free

#ifdef __cplusplus
}
#endif

#endif /* KISS_FFTR_H */
