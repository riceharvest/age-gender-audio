"""Reference numpy implementation of the trained heads (R2 recipe), used to cross-check the ONNX graph.

Input is the mean-pooled last hidden state of frozen facebook/wav2vec2-base (768-d). Heads: per-feature
standardisation (train mean/std) -> Linear(768,256) -> GELU -> age Linear(256,1) [age/100, L1-trained on
decade-bucket midpoints] and gender Linear(256,3) [logits female, male, child]. The child logit is never a
training target: it is untrained and must not be read as a prediction.
"""
from __future__ import annotations

import math
import os

import numpy as np

HEAD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "head.npz")
KEYS = ("mu", "sd", "w1", "b1", "wa", "ba", "wg", "bg")
_erf = np.frompyfunc(math.erf, 1, 1)


def load_head(path: str = HEAD_FILE) -> dict:
    d = np.load(path)
    h = {k: np.asarray(d[k], dtype=np.float32) for k in KEYS}
    shapes = {"mu": (768,), "sd": (768,), "w1": (256, 768), "b1": (256,), "wa": (1, 256), "ba": (1,),
              "wg": (3, 256), "bg": (3,)}
    for k, s in shapes.items():
        if h[k].shape != s:
            raise ValueError(f"head.npz[{k}] has shape {h[k].shape}, expected {s}")
    return h


def gelu(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)).astype(np.float64))).astype(np.float32)


def forward(emb: np.ndarray, head: dict | None = None):
    """emb [N,768] pooled backbone features -> (age [N,1] in 0..1 scale, gender logits [N,3])."""
    h = head if head is not None else load_head()
    emb = np.atleast_2d(np.asarray(emb, dtype=np.float32))
    z = gelu(((emb - h["mu"]) / h["sd"]) @ h["w1"].T + h["b1"])
    return z @ h["wa"].T + h["ba"], z @ h["wg"].T + h["bg"]
