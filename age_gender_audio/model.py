"""ONNX Runtime inference: age and gender from speech.

Output convention follows the incumbent audeering/wav2vec2-large-robust-24-ft-age-gender
(`forward -> (embedding, age, gender)`): age on a 0..1 scale (x100 = years), gender as probabilities
in the order [female, male, child]. Differences, stated plainly:
  * the child entry is NOT predicted (it is never a training target); use argmax over [female, male] only;
  * age is trained against decade-bucket midpoints, so it has decade resolution (floor ~2.5 yr MAE);
  * embedding is the 768-d mean-pooled last hidden state of the frozen wav2vec2-base backbone.
The graph takes one clip at a time (batch 1): wav2vec2-base group-norm is length dependent, so padding changes outputs.
"""
from __future__ import annotations

import numpy as np

from . import audio

LABELS = ("female", "male", "child")


def _softmax(l: np.ndarray) -> np.ndarray:
    e = np.exp(l - l.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class AgeGenderModel:
    def __init__(self, onnx_path: str, providers=None, intra_op_threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(onnx_path, so, providers=providers or ["CPUExecutionProvider"])
        self.onnx_path = onnx_path

    def run_raw(self, x: np.ndarray):
        """x: preprocessed waveform [T] -> (embedding [768], age [1] in 0..1, gender logits [3])."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1 or x.size < 400:
            raise ValueError("expected a 1-D preprocessed waveform of at least 400 samples (25 ms at 16 kHz)")
        emb, age, lg = self.session.run(["embedding", "age", "gender_logits"], {"audio": x[None]})
        return emb[0], age[0], lg[0]

    def forward(self, wav: np.ndarray, sr: int = audio.SAMPLE_RATE):
        """Raw waveform at rate sr -> (embedding [768], age [1] 0..1, gender probs [3] female/male/child)."""
        emb, age, lg = self.run_raw(audio.preprocess(wav, sr))
        return emb, age, _softmax(lg)

    __call__ = forward

    def predict(self, src) -> dict:
        """Path / bytes -> {'age_years', 'gender', 'gender_probs'}; gender = argmax over [female, male] only."""
        emb, age, lg = self.run_raw(audio.load_audio(src))
        p = _softmax(lg)
        return {"age_years": float(age[0]) * 100.0,
                "gender": LABELS[int(np.argmax(lg[:2]))],
                "gender_probs": {k: float(v) for k, v in zip(LABELS, p)}}
