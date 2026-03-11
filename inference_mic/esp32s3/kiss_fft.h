/*
 * Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the author nor the names of any contributors may be used to
 *    endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.
 */

#ifndef KISS_FFT_H
#define KISS_FFT_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float kiss_fft_scalar;

typedef struct {
    kiss_fft_scalar r;
    kiss_fft_scalar i;
} kiss_fft_cpx;

typedef struct kiss_fft_state * kiss_fft_cfg;

/*
 * kiss_fft_alloc
 *
 * Initialize a FFT (or IFFT) algorithm's cfg/state buffer.
 *
 * typical usage:      kiss_fft_cfg mycfg = kiss_fft_alloc(1024, 0, NULL, NULL);
 *
 * The return value from fft_alloc is a cfg buffer used in fft:
 *      kiss_fft(mycfg, *, *);
 *
 * If lenmem is NULL, then kiss_fft_alloc will allocate a structure
 * and return it.  The structure MUST be freed by the SAME method that was
 * used to allocate it (kiss_fft_free if internally allocated, or user-supplied
 * if the memory was user-supplied).
 *
 * If lenmem is not NULL and mem is not NULL: user allocated storage
 * If lenmem is not NULL and mem is NULL: returns the required size in *lenmem
 * inverse_fft: 0 = forward FFT, 1 = inverse FFT
 */
kiss_fft_cfg kiss_fft_alloc(int nfft, int inverse_fft,
                             void * mem, size_t * lenmem);

/*
 * kiss_fft(cfg, fin, fout)
 *
 * Perform an FFT on a complex input buffer.
 * for a forward FFT:
 *  fin  should be  f[0], f[1], ..., f[nfft-1]
 *  fout will be    F[0], F[1], ..., F[nfft-1]
 * NOTE: fout and fin may be the same buffer, but the results may not be exactly
 * the same as out-of-place.  fin is not modified unless fin==fout.
 */
void kiss_fft(kiss_fft_cfg cfg, const kiss_fft_cpx * fin, kiss_fft_cpx * fout);

/*
 * A more generic version of the above function.
 * Input and output are complex buffers with the following strides:
 *   fin  is sampled every fin_stride  elements
 *   fout is written every 1 element
 */
void kiss_fft_stride(kiss_fft_cfg cfg, const kiss_fft_cpx * fin,
                     kiss_fft_cpx * fout, int fin_stride);

/* If kiss_fft_alloc allocated a buffer, it is one contiguous
   buffer and can be freed with free() */
#define kiss_fft_free free

/* Cleans up some memory that is used by the fft */
void kiss_fft_cleanup(void);

/*
 * Returns the smallest integer k such that k>=n and k has only small
 * prime factors.
 */
int kiss_fft_next_fast_size(int n);

/* For real FFTs, this is the size that the complex FFT's nfft should be */
#define kiss_fftr_next_fast_size_real(n) \
        (kiss_fft_next_fast_size( ((n)+1)>>1)<<1)

#ifdef __cplusplus
}
#endif

#endif /* KISS_FFT_H */
