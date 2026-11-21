"""Audio front end, identical to the recipe the heads were trained with (R2 extract.py).

mono -> 16 kHz -> first 10 s -> zero-mean / unit-variance. Pure numpy; file decoding needs
`soundfile` (and `scipy` for resampling), imported lazily so the module imports without them.
"""
from __future__ import annotations

import io
from math import gcd

import numpy as np

SAMPLE_RATE = 16000
MAX_SECONDS = 10
MAX_SAMPLES = MAX_SECONDS * SAMPLE_RATE


def to_mono_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """x: float array [T] or [T, C] at rate sr -> float32 mono at 16 kHz."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(1)
    if x.ndim != 1:
        raise ValueError(f"expected [T] or [T, C] audio, got shape {x.shape}")
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = gcd(int(sr), SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return x


def normalize(x: np.ndarray) -> np.ndarray:
    """First 10 s, then zero mean / unit variance (the wav2vec2 feature-extractor setting)."""
    x = np.asarray(x, dtype=np.float32)[:MAX_SAMPLES]
    if x.size == 0:
        raise ValueError("empty audio")
    return ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)


def preprocess(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Raw waveform at rate sr -> model input [T'] float32 (T' <= 160000)."""
    return normalize(to_mono_16k(x, sr))


def load_audio(src) -> np.ndarray:
    """Path, file object or bytes (any format libsndfile reads, e.g. wav/flac/ogg/mp3) -> model input."""
    import soundfile as sf

    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    return preprocess(x, sr)
