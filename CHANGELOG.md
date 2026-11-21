# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).
Any change to an existing API's output is a major/minor decision (lab policy: decision card for the owner, default keep the old output).

## [Unreleased]

## [0.1.0] - TBD

### Added
- Initial scaffold: package skeleton, CLI, tests, CI, hygiene hooks, honest-claims README.
- Model code: frozen facebook/wav2vec2-base backbone + trained age head and 3-logit gender head (R2 recipe, seed 0), ONNX Runtime wrapper (`AgeGenderModel`, output order of audeering/wav2vec2-large-robust-24-ft-age-gender: embedding, age 0..1, gender probabilities [female, male, child]; the child entry is not predicted), CLI.
- int8 ONNX export that quantizes only transformer-layer weight matmuls (60 MatMulInteger nodes, asserted; conv front end, fc2 and heads stay fp32).
- R4 release-gate set (`r4/`): 1200 speaker-disjoint CV17-en clips, list sha256 pinned before any model number; draw, scoring and gate scripts.