import hashlib
import io

import numpy as np
import pytest

from age_gender_audio import audio, heads


def test_normalize_zero_mean_unit_var_and_10s_crop():
    x = np.random.default_rng(0).normal(3.0, 5.0, 16000 * 12).astype(np.float32)
    y = audio.normalize(x)
    assert y.shape == (audio.MAX_SAMPLES,)
    assert abs(float(y.mean())) < 1e-4 and abs(float(y.std()) - 1.0) < 1e-3
    assert y.dtype == np.float32


def test_normalize_rejects_empty():
    with pytest.raises(ValueError):
        audio.normalize(np.zeros(0, np.float32))


def test_stereo_is_averaged_to_mono():
    st = np.stack([np.ones(100, np.float32), 3 * np.ones(100, np.float32)], 1)
    assert np.allclose(audio.to_mono_16k(st, 16000), 2.0)


def test_resample_length():
    pytest.importorskip("scipy")
    x = np.sin(np.linspace(0, 100, 48000)).astype(np.float32)  # 1 s at 48 kHz
    assert audio.to_mono_16k(x, 48000).shape == (16000,)


def test_load_audio_from_wav_bytes():
    sf = pytest.importorskip("soundfile")
    buf = io.BytesIO()
    sf.write(buf, np.random.default_rng(1).normal(size=16000).astype(np.float32) * 0.1, 16000, format="WAV")
    y = audio.load_audio(buf.getvalue())
    assert y.shape == (16000,) and abs(float(y.mean())) < 1e-4


def test_head_file_is_the_r2_seed0_head():
    # sha256 of the packed head; its source is R2 head_seed0.pt (sha256 cfb79ff2956c...923afb), packed by scripts/pack_head.py
    h = hashlib.sha256(open(heads.HEAD_FILE, "rb").read()).hexdigest()
    assert h == "5bf3380631ee96c5ff53dff656874dd199f8bcd52846ec460b8aa056d636cc5a"


def test_head_shapes_and_forward():
    h = heads.load_head()
    assert h["w1"].shape == (256, 768) and h["wg"].shape == (3, 256)
    emb = np.random.default_rng(2).normal(size=(4, 768)).astype(np.float32)
    age, lg = heads.forward(emb, h)
    assert age.shape == (4, 1) and lg.shape == (4, 3)
    assert np.isfinite(age).all() and np.isfinite(lg).all()


def test_gelu_reference_values():
    # exact (erf) GELU: gelu(0)=0, gelu(1)=0.8413447, gelu(-1)=-0.1586553
    g = heads.gelu(np.array([0.0, 1.0, -1.0], np.float32))
    assert np.allclose(g, [0.0, 0.8413447, -0.1586553], atol=1e-6)
