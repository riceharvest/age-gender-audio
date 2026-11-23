# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).
Any change to an existing API's output is a major/minor decision (lab policy: decision card for the owner, default keep the old output).

## [0.1.1] - 2026-11-23

Hotfix on v0.1.0 (patch). Runtime API, CLI, ONNX I/O names and head weights are unchanged (API probe: IDENTICAL, [`probe/api_probe_0.1.1.md`](https://github.com/riceharvest/age-gender-audio/blob/main/probe/api_probe_0.1.1.md)). No new held-out run was made for 0.1.1; the numbers below are the 0.1.0 measurements, scoped as stated.

### Fixed
- `export` extra pins `transformers>=4.57,<5` (5.x crashes the wav2vec2 export with an attention IndexError) and `export_fp32` fails fast with `export needs transformers >=4.57,<5, found <ver>`. Tested: transformers 4.57.6, torch 2.15.0, onnx 1.23.2, onnxruntime 1.32.0.
- Documented export sha re-recorded from that environment: fp32 `3a28b86f...1bc4` (was `d9d7d004...` from transformers 5.5.4, not reproducible); int8 `71436e54...65df` unchanged. Head weights `age_gender_audio/data/head.npz` unchanged (sha256 `5bf33806...cc5a`; `git diff --quiet v0.1.0 HEAD -- age_gender_audio/data` is clean).
- Checks on this release: `sha256sum` of the pinned-environment export `age_gender_fp32.onnx` = `3a28b86f...1bc4` (378,448,514 B) and `age_gender_int8.onnx` = `71436e54...65df` (209,118,507 B), the same int8 file as 0.1.0. Re-export at the pin was run twice with identical sha256 ([`r4/export_pin_result.md`](https://github.com/riceharvest/age-gender-audio/blob/main/r4/export_pin_result.md)).

### Scope of the numbers in this release
- **int8 (shipped file):** the held-out R4 numbers below were measured on int8 sha256 `71436e54...65df`, the same file as 0.1.1. They carry over; no re-measure.
- **fp32:** the R4 fp32 row was measured on the old fp32 build (`d9d7d004...`, transformers 5.5.4). The 0.1.1 fp32 file (`3a28b86f...`) was not scored on R4. Do not quote the fp32 row as the 0.1.1 fp32 result.
- No wording change from 0.1.0: "level with" the incumbent, never "beats".

### Evidence carried from 0.1.0 (scoped)
- Held-out R4, n=1200 CV17-en speakers, list sha256 `90d1c69b...fe07`, int8 vs incumbent: gender 0.9583 vs 0.9492 (+0.92 pp, CI [-0.17, +2.00]); age MAE 9.192 vs 9.741 yr (-0.550 yr, CI [-1.096, +0.024]). Gate PASS. Source: [`r4/result.md`](https://github.com/riceharvest/age-gender-audio/blob/main/r4/result.md), [`r4/gate_r4.json`](https://github.com/riceharvest/age-gender-audio/blob/main/r4/gate_r4.json) (`arm: int8`, `gate_pass: true`).
- Dev-set R2 screen (n=600; the lab result calls these clips held-out for the heads, but they were also used to pick the int8 recipe, so the README labels them dev-set): ours behind the incumbent on both seeds (gender -0.33 / -0.50 pp; age MAE +0.571 / +0.523 yr). Losses are kept in the README table.
- API probe: IDENTICAL against v0.1.0 ([`probe/api_probe_0.1.1.md`](https://github.com/riceharvest/age-gender-audio/blob/main/probe/api_probe_0.1.1.md)).

### Known limits
- Same as 0.1.0 (child never predicted; CV17-en only; age at decade-bucket resolution). 0.1.1 adds no measurement of any other language, age group or audio condition.

## [0.1.0] - 2026-11-21

### Added
- Initial scaffold: package skeleton, CLI, tests, CI, hygiene hooks, honest-claims README.
- Model code: frozen facebook/wav2vec2-base backbone + trained age head and 3-logit gender head (R2 recipe, seed 0), ONNX Runtime wrapper (`AgeGenderModel`, output order of audeering/wav2vec2-large-robust-24-ft-age-gender: embedding, age 0..1, gender probabilities [female, male, child]; the child entry is not predicted), CLI.
- int8 ONNX export that quantizes only transformer-layer weight matmuls (60 MatMulInteger nodes, asserted; conv front end, fc2 and heads stay fp32).
- R4 release-gate set (`r4/`): 1200 speaker-disjoint CV17-en clips, list sha256 pinned before any model number; draw, scoring and gate scripts.

### Evidence (scoped; full tables and CIs in the [README](https://github.com/riceharvest/age-gender-audio/blob/main/README.md#results-scoped-010))
- Held-out R4, n=1200 CV17-en speakers, list sha256 `90d1c69b...fe07`: int8 gender 0.9583 vs incumbent 0.9492 (+0.92 pp, CI [-0.17, +2.00]); age MAE 9.192 vs 9.741 yr (-0.550 yr, CI [-1.096, +0.024]). Pre-registered gate PASS. Wording: level with the incumbent, not "beats" (CIs include 0). Source: [`r4/result.md`](https://github.com/riceharvest/age-gender-audio/blob/main/r4/result.md), [`r4/gate_r4.json`](https://github.com/riceharvest/age-gender-audio/blob/main/r4/gate_r4.json).
- Dev-set R2 screen, n=600 (spent): ours was behind the incumbent on both seeds (gender 0.9467 / 0.9450 vs 0.9500; age MAE 10.383 / 10.336 vs 9.812 yr), within the pre-registered margin. Source: lab repo `results/age-gender-audio/r2/result.md`, review PASS.
- int8 vs fp32 parity: held-out gender -0.17 pp, age +0.015 yr (PASS).

### Known limits
- `child` probability is never predicted (top-class child 0/1200 on R4; the incumbent 10/1200).
- Trained and tested on Common Voice 17 English only; age is regressed against decade-bucket midpoints. Test sets are stratified, not population draws.
- API/CLI surface baseline: [`probe/api_probe_0.1.0.md`](https://github.com/riceharvest/age-gender-audio/blob/main/probe/api_probe_0.1.0.md) (first release; no earlier release to compare).
