#!/usr/bin/env python3
"""Decode a clip list's audio into the npz the lab's R2 incumbent rescore (rescore.py run) reads:
s{i} = first 10 s, 16 kHz mono, zero-mean/unit-var (same as age_gender_audio.audio.load_audio); f{i} = full-length, normalised.
Usage: make_audio_npz.py LIST.json (--audio-dir DIR | --raw RAW_DIR) OUT.npz
"""
import argparse, io, json, os, sys
import numpy as np
import soundfile as sf

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "..")]
from age_gender_audio import audio

ap = argparse.ArgumentParser()
ap.add_argument("lst"); ap.add_argument("out"); ap.add_argument("--audio-dir"); ap.add_argument("--raw")
a = ap.parse_args()
items = json.load(open(a.lst))
if a.audio_dir:
    blobs = [open(os.path.join(a.audio_dir, f"{i:04d}.mp3"), "rb").read() for i in range(len(items))]
else:
    from parquet_clips import load_blobs
    blobs = load_blobs(items, a.raw)
d = {}
for i, b in enumerate(blobs):
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    w = audio.to_mono_16k(x, sr)
    d[f"s{i}"] = audio.normalize(w)
    d[f"f{i}"] = ((w - w.mean()) / np.sqrt(w.var() + 1e-7)).astype(np.float32)
np.savez(a.out, **d)
print("wrote", a.out, len(items))
