# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).
Any change to an existing API's output is a major/minor decision (lab policy: decision card for the owner, default keep the old output).

## [Unreleased]

### Fixed
- `export` extra pins `transformers>=4.57,<5` (5.x crashes the wav2vec2 export with an attention IndexError) and `export_fp32` fails fast with `export needs transformers >=4.57,<5, found <ver>`. Tested: transformers 4.57.6, torch 2.15.0, onnx 1.23.2, onnxruntime 1.32.0.
- Documented export sha re-recorded from that environment: fp32 `3a28b86f...1bc4` (was `d9d7d004...` from transformers 5.5.4, not reproducible); int8 `71436e54...65df` unchanged. Model outputs and heads unchanged.

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
