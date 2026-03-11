/*
 * mel_spectrogram.h  —  Mel spectrogram computation for ESP32-S3
 *
 * Matches torchaudio.transforms.MelSpectrogram with:
 *   sample_rate = 44100
 *   n_fft       = 2048
 *   hop_length  = 512
 *   n_mels      = 64
 *   window      = Hann
 *   center      = True  (n_fft//2 = 1024 zeros prepended & appended)
 *   power       = 2.0   (power spectrogram)
 *   norm        = None
 *   mel_scale   = "htk"
 *
 * AmplitudeToDB:
 *   stype  = "power"  →  10 * log10(x)
 *   top_db = None     →  no clipping
 *
 * Output shape: [N_MELS][T_FRAMES]  (431 frames for 5 s @ 44100 Hz)
 *
 * Dependencies:
 *   kiss_fftr.h / kiss_fftr.c — real FFT
 *   mel_filterbank.h          — precomputed 64×1025 filterbank
 *                               (run tools/gen_mel_filterbank.py first)
 */

#pragma once

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "kiss_fftr.h"
#include "mel_filterbank.h"

/* ── Parameters (must match onnx_to_tflite.py tracing parameters) ────── */
#define MEL_SAMPLE_RATE  44100
#define MEL_N_FFT        2048
#define MEL_HOP_LENGTH   512
#define MEL_N_MELS_DEF   64       /* also declared in mel_filterbank.h    */
#define MEL_T_FRAMES     431
#define MEL_PAD          (MEL_N_FFT / 2)    /* 1024, center=True padding  */
#define MEL_N_FREQS_DEF  (MEL_N_FFT / 2 + 1)  /* 1025                    */

/* ── Public API ─────────────────────────────────────────────────────────

   compute_mel_spectrogram(
       samples,         float array of MEL_T_FRAMES resampled samples
       n_samples,       length of samples (must be >= 220500 for T=431)
       out_db           [MEL_N_MELS_DEF][MEL_T_FRAMES] output in dB
   )

   Returns 0 on success, -1 on allocation failure.
   The caller owns the output buffer; it must be allocated before the call.
 */

static int compute_mel_spectrogram(
    const float * __restrict__ samples,
    int    n_samples,
    float  out_db[MEL_N_MELS_DEF][MEL_T_FRAMES]
)
{
    const int nfft     = MEL_N_FFT;
    const int hop      = MEL_HOP_LENGTH;
    const int pad      = MEL_PAD;
    const int n_freqs  = MEL_N_FREQS_DEF;
    const int n_mels   = MEL_N_MELS_DEF;
    const int t_frames = MEL_T_FRAMES;

    /* ── Precompute Hann window ──────────────────────────────────────── */
    static float hann[MEL_N_FFT];
    static int   hann_ready = 0;
    if (!hann_ready) {
        for (int i = 0; i < nfft; i++)
            hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (nfft - 1)));
        hann_ready = 1;
    }

    /* ── Allocate KissFFT real-FFT config ───────────────────────────── */
    kiss_fftr_cfg fft_cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    if (!fft_cfg) return -1;

    /* ── Frame buffers ───────────────────────────────────────────────── */
    static kiss_fft_scalar  frame_buf[MEL_N_FFT];
    static kiss_fft_cpx     freq_buf[MEL_N_FFT / 2 + 1];

    /* ── padded_length = n_samples + 2*pad
          With center=True: the first frame is centred on sample 0,
          so we prepend and append pad zeros.                             */
    int padded_len = n_samples + 2 * pad;

    for (int t = 0; t < t_frames; t++) {
        int frame_start = t * hop;   /* start offset in the PADDED signal */

        /* Build windowed frame: map padded index → sample index         */
        for (int i = 0; i < nfft; i++) {
            int idx = frame_start + i;   /* index in padded signal       */
            float s;
            if (idx < pad || idx >= pad + n_samples) {
                s = 0.0f;               /* zero-padding region            */
            } else {
                s = samples[idx - pad];
            }
            frame_buf[i] = s * hann[i];
        }

        /* FFT */
        kiss_fftr(fft_cfg, frame_buf, freq_buf);

        /* Power spectrum |X[k]|^2 then apply mel filterbank */
        for (int m = 0; m < n_mels; m++) {
            float energy = 0.0f;
            for (int k = 0; k < n_freqs; k++) {
                float pw = freq_buf[k].r * freq_buf[k].r
                         + freq_buf[k].i * freq_buf[k].i;
                energy += MEL_FILTERBANK[m][k] * pw;
            }
            /* AmplitudeToDB: 10 * log10(max(energy, 1e-10)) */
            out_db[m][t] = 10.0f * log10f(fmaxf(energy, 1e-10f));
        }
    }

    kiss_fftr_free(fft_cfg);
    return 0;
}

/*
 * dc_remove_normalize:
 *   Remove DC bias and normalize samples to [-1, 1] in-place.
 *   Call this before compute_mel_spectrogram.
 */
static void dc_remove_normalize(float * samples, int n)
{
    if (n == 0) return;

    /* mean */
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += samples[i];
    mean /= n;

    /* subtract mean */
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        samples[i] -= mean;
        float a = fabsf(samples[i]);
        if (a > max_abs) max_abs = a;
    }

    /* normalize */
    if (max_abs > 0.0f) {
        float scale = 1.0f / max_abs;
        for (int i = 0; i < n; i++) samples[i] *= scale;
    }
}
