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
