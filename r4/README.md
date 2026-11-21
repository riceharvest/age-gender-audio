# R4 release-gate set (0.1.0)

`r4_clips.json`: 1200 Common Voice 17 English clips, one per `client_id` (1200 distinct speakers), drawn by
`scripts/draw_r4.py` from the speakers reserved in R2 prep (`int(sha256('agr2:'+client_id)[:8],16) % 10 == 1`).
R2 trained on hash >= 2 and screened on hash == 0, so no R4 speaker was ever trained on, used for selection or scored.
Draw facts are in `r4_draw_report.json` (pool 1233 labelled speakers over en train shards 0000-0089; K=1 clip per speaker;
cell cap C=393 on (age bucket, gender); seed 20261124; client_id overlap with the R2 train list 0, with the R2 screen list 0).
The list pins each clip's audio by sha256 (`audio_sha256`).

**The list was hashed before any model number existed on it:**
`90d1c69b0405fb75bb15916362cb7e914d360b6775502cd00e848a392967fe07  r4_clips.json` (`sha256sum -c r4_clips.sha256`).

Composition (NOT a population figure; stratified like the R2 screen, capped at the largest cell): male 917 / female 283;
age buckets 15:171 25:499 35:236 45:149 55:76 65:47 75:19 85:3.

## Reproduce
```
scripts/fetch_r4.sh <data>/raw                                  # CC0 shards
python scripts/draw_r4.py <data>/raw <r2-prep-dir> r4 <data>/r4_audio   # needs the R2 prep lists (lab repo, regenerable with R2 prep.py)
sha256sum -c r4/r4_clips.sha256
python -m age_gender_audio.export OUTDIR                              # fp32 + int8 ONNX, asserts the int8 graph
python scripts/score_onnx.py r4/r4_clips.json FP32.onnx INT8.onnx ours.json --audio-dir <data>/r4_audio
python scripts/make_audio_npz.py r4/r4_clips.json r4.npz --audio-dir <data>/r4_audio   # input of the incumbent rescore (lab R2 rescore.py run)
python scripts/gate_r4.py r4/r4_clips.json incumbent.json ours.json gate.json
```

## Build-time evidence (spent set, not the R4 clips)
`scripts/parity.py` on the R2 screen600 (already read in R2): fp32 ONNX vs R2 torch predictions max |age diff| 0.055 yr, max |logit diff| 0.016,
gender argmax agreement 1.0; int8 vs fp32: gender accuracy 0.9467 -> 0.9450 (-0.17 pp), age MAE 10.383 -> 10.472 yr (+0.089 yr),
gender argmax agreement 0.995 (`int8_parity_screen600_spent_set.json`). The int8 recipe was chosen on this set (see `age_gender_audio/export.py` docstring).
Export environment: torch 2.6.0+cpu, transformers 5.5.4, onnx 1.23.2, onnxruntime 1.32.0 (venv a local venv). fp32 ONNX sha256 d9d7d004142f9f0900d55e78a9e7e16495ab5f3a2c82a120c4c277692888afc8 (378,448,513 B, reproduced bit-identical on a second export); int8 sha256 71436e545d79771b386004d2932290a410f77e1f4efca2edc118e3cc079965df (209,118,507 B).
