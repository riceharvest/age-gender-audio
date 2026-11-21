#!/usr/bin/env python3
"""Score a clip list with the fp32 and int8 ONNX graphs (CPU). Writes one row per clip, list order.
Audio comes from --audio-dir (NNNN.mp3, row order of the list; each file's sha256 must match the list's audio_sha256 when present)
or from --raw (CV17 parquet shards, rows by shard/row).
Usage: score_onnx.py LIST.json FP32.onnx INT8.onnx OUT.json (--audio-dir DIR | --raw RAW_DIR) [--threads T]
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "..")]
from age_gender_audio import audio
from age_gender_audio.model import AgeGenderModel

ap = argparse.ArgumentParser()
ap.add_argument("lst"); ap.add_argument("fp32"); ap.add_argument("int8"); ap.add_argument("out")
ap.add_argument("--audio-dir"); ap.add_argument("--raw"); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
items = json.load(open(a.lst))
if a.limit:
    items = items[:a.limit]
if a.audio_dir:
    blobs = []
    for i, it in enumerate(items):
        b = open(os.path.join(a.audio_dir, f"{i:04d}.mp3"), "rb").read()
        if "audio_sha256" in it:
            assert hashlib.sha256(b).hexdigest() == it["audio_sha256"], f"audio {i} does not match the list"
        blobs.append(b)
else:
    from parquet_clips import load_blobs
    blobs = load_blobs(items, a.raw)
m32, m8 = AgeGenderModel(a.fp32, intra_op_threads=a.threads), AgeGenderModel(a.int8, intra_op_threads=a.threads)
rows, t0 = [], time.time()
for i, (it, b) in enumerate(zip(items, blobs)):
    x = audio.load_audio(b)
    _, a32, l32 = m32.run_raw(x)
    _, a8, l8 = m8.run_raw(x)
    rows.append({"shard": it["shard"], "row": it["row"], "age_fp32": float(a32[0]) * 100, "age_int8": float(a8[0]) * 100,
                 "logits_fp32": [float(v) for v in l32], "logits_int8": [float(v) for v in l8]})
    if i % 100 == 0:
        print(i, f"{time.time()-t0:.0f}s", flush=True)
json.dump(rows, open(a.out, "w"))
print("wrote", a.out, len(rows), f"{time.time()-t0:.0f}s")
