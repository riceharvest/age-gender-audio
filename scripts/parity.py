#!/usr/bin/env python3
"""Packaging check on an ALREADY-SPENT set (the R2 screen600, never the R4 clips):
  (a) fp32 ONNX vs the R2 torch predictions saved for the same head (screen600_pred_seed0.json),
  (b) int8 ONNX vs fp32 ONNX (gender 2-way agreement / accuracy delta, age MAE delta vs decade-bucket labels).
Usage: parity.py FP32.onnx INT8.onnx SCREEN600.json RAW_DIR R2_PRED.json OUT.json [--limit N] [--threads T]
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from parquet_clips import load_blobs
from age_gender_audio import audio
from age_gender_audio.model import AgeGenderModel

fp32, int8, scr, raw, pred, out = sys.argv[1:7]
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
limit, thr = int(arg("--limit", 0)) or None, int(arg("--threads", 4))
items = json.load(open(scr))[:limit]
ref = json.load(open(pred))[:len(items)]
assert all(r["shard"] == p["shard"] and r["row"] == p["row"] for r, p in zip(items, ref))
blobs = load_blobs(items, raw)
m32, m8 = AgeGenderModel(fp32, intra_op_threads=thr), AgeGenderModel(int8, intra_op_threads=thr)
rows, t0 = [], time.time()
for i, (r, b) in enumerate(zip(items, blobs)):
    x = audio.load_audio(b)
    e32, a32, g32 = m32.run_raw(x)
    e8, a8, g8 = m8.run_raw(x)
    rows.append((a32[0] * 100, a8[0] * 100, g32, g8))
    if i % 100 == 0:
        print(i, f"{time.time()-t0:.0f}s", flush=True)
age_true = np.array([r["age"] for r in items], float)
g_true = np.array([0 if r["gender"] == "female" else 1 for r in items])
a32, a8 = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
G32, G8 = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
a_ref = np.array([p["age_pred"] for p in ref]); G_ref = np.array([p["gender_logits"] for p in ref])
acc = lambda G: float((G[:, :2].argmax(1) == g_true).mean())
res = {"n": len(items), "set": "R2 screen600 (already spent; not the R4 clips)",
       "fp32_vs_r2torch": {"max_abs_age_diff_yr": float(np.abs(a32 - a_ref).max()),
                           "max_abs_logit_diff": float(np.abs(G32 - G_ref).max()),
                           "gender_argmax_agree": float((G32[:, :2].argmax(1) == G_ref[:, :2].argmax(1)).mean())},
       "int8_vs_fp32": {"gender_acc_fp32": acc(G32), "gender_acc_int8": acc(G8),
                        "gender_acc_delta_pp": 100 * (acc(G8) - acc(G32)),
                        "gender_argmax_agree": float((G32[:, :2].argmax(1) == G8[:, :2].argmax(1)).mean()),
                        "age_mae_fp32": float(np.abs(a32 - age_true).mean()), "age_mae_int8": float(np.abs(a8 - age_true).mean()),
                        "age_mae_delta_yr": float(np.abs(a8 - age_true).mean() - np.abs(a32 - age_true).mean()),
                        "max_abs_age_diff_yr": float(np.abs(a32 - a8).max()), "mean_abs_age_diff_yr": float(np.abs(a32 - a8).mean())}}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
