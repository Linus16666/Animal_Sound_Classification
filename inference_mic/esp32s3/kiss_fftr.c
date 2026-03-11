/*
 * Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "kiss_fftr.h"
#include "_kiss_fft_guts.h"

struct kiss_fftr_state {
    kiss_fft_cfg substate;
    kiss_fft_cpx * tmpbuf;
    kiss_fft_cpx * super_twiddles;
};

kiss_fftr_cfg kiss_fftr_alloc(int nfft, int inverse_fft,
                               void * mem, size_t * lenmem)
{
    int i;
    kiss_fftr_cfg st = NULL;
    size_t subsize;
    size_t memneeded;

    if (nfft & 1) {
        /* nfft must be even */
        return NULL;
    }

    /* get substate size */
    kiss_fft_alloc(nfft / 2, inverse_fft, NULL, &subsize);
    memneeded = sizeof(struct kiss_fftr_state) +
                subsize +
                sizeof(kiss_fft_cpx) * (nfft / 2) +  /* tmpbuf */
                sizeof(kiss_fft_cpx) * (nfft / 2);   /* super_twiddles */

    if (lenmem == NULL) {
        st = (kiss_fftr_cfg)malloc(memneeded);
    } else {
        if (mem != NULL && *lenmem >= memneeded)
            st = (kiss_fftr_cfg)mem;
        *lenmem = memneeded;
    }

    if (!st) return NULL;

    {
        uint8_t * ptr = (uint8_t *)(st + 1);
        st->substate    = (kiss_fft_cfg)ptr;
        ptr += subsize;
        st->tmpbuf      = (kiss_fft_cpx *)ptr;
        ptr += sizeof(kiss_fft_cpx) * (nfft / 2);
        st->super_twiddles = (kiss_fft_cpx *)ptr;
    }

    kiss_fft_alloc(nfft / 2, inverse_fft, st->substate, &subsize);

    for (i = 0; i < nfft / 2; ++i) {
        double phase = -M_PI * ((double)i / nfft + 0.5);
        if (inverse_fft)
            phase = -phase;
        kf_cexp(st->super_twiddles + i, (float)phase);
    }
    return st;
}

void kiss_fftr(kiss_fftr_cfg st, const kiss_fft_scalar * timedata,
               kiss_fft_cpx * freqdata)
{
    /* input buffer timedata is stored row-wise */
    int k, ncfft;
    kiss_fft_cpx fpnk, fpk, f1k, f2k, tw, tdc;
    ncfft = st->substate->nfft;

    /* perform the FFT on the complex-valued buffer */
    /* The real input is treated as ncfft complex points interleaved */
    kiss_fft(st->substate, (const kiss_fft_cpx *)timedata, st->tmpbuf);

    tdc.r = st->tmpbuf[0].r;
    tdc.i = st->tmpbuf[0].i;

    freqdata[0].r     = tdc.r + tdc.i;
    freqdata[ncfft].r = tdc.r - tdc.i;
    freqdata[ncfft].i = freqdata[0].i = 0;

    for (k = 1; k <= ncfft / 2; ++k) {
        fpk  = st->tmpbuf[k];
        fpnk.r =   st->tmpbuf[ncfft - k].r;
        fpnk.i = - st->tmpbuf[ncfft - k].i;

        C_ADD(f1k, fpk, fpnk);
        C_SUB(f2k, fpk, fpnk);
        C_MUL(tw, f2k, st->super_twiddles[k - 1]);

        freqdata[k].r     = HALF_OF(f1k.r + tw.r);
        freqdata[k].i     = HALF_OF(f1k.i + tw.i);
        freqdata[ncfft - k].r = HALF_OF(f1k.r - tw.r);
        freqdata[ncfft - k].i = HALF_OF(-(f1k.i - tw.i));
    }
}

void kiss_fftri(kiss_fftr_cfg st, const kiss_fft_cpx * freqdata,
                kiss_fft_scalar * timedata)
{
    /* NOTE: This is NOT the inverse transform by default — caller must divide by nfft */
    int k, ncfft;
    ncfft = st->substate->nfft;

    st->tmpbuf[0].r = freqdata[0].r + freqdata[ncfft].r;
    st->tmpbuf[0].i = freqdata[0].r - freqdata[ncfft].r;

    for (k = 1; k <= ncfft / 2; ++k) {
        kiss_fft_cpx fk, fnkc, fek, fok, tmp;

        fk   = freqdata[k];
        fnkc.r =  freqdata[ncfft - k].r;
        fnkc.i = -freqdata[ncfft - k].i;

        C_ADD(fek, fk, fnkc);
        C_SUB(tmp, fk, fnkc);
        C_MUL(fok, tmp, st->super_twiddles[k - 1]);

        st->tmpbuf[k].r     = fek.r - fok.i;
        st->tmpbuf[k].i     = fek.i + fok.r;
        st->tmpbuf[ncfft - k].r = fek.r + fok.i;
        st->tmpbuf[ncfft - k].i = fok.r - fek.i;
    }
    kiss_fft(st->substate, st->tmpbuf, (kiss_fft_cpx *)timedata);
}
