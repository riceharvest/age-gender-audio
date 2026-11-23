IDENTICAL (ship). Probe of 0.1.1 (HEAD 0cddd88, code 566979a) against the previous release 0.1.0 (tag v0.1.0). No 0.0.0 release exists (see api_probe_0.1.0.md).

Method: `git diff v0.1.0 HEAD -- age_gender_audio pyproject.toml scripts` plus a signature dump of every public module (inspect.signature, both trees, default paths normalised).

- Changed files in the package: `__init__.py` (`__version__` 0.1.0 -> 0.1.1), `export.py`, `pyproject.toml` (version; `export` extra `transformers` -> `transformers>=4.57,<5`).
- Unchanged: `cli.py`, `model.py`, `heads.py`, `data/` (head weights), `scripts/`. So `AgeGenderModel`, `run_raw`, `forward`, `predict`, ONNX I/O names, the CLI flags and the JSON line format are byte-identical in source.
- Only signature diff: `export.assert_transformers_version(found=None) -> str` is new (additive, export-only helper). `export_fp32` and `build_torch_module` keep their signatures; `export_fp32` now raises AssertionError if transformers is outside >=4.57,<5 (build-time guard, not runtime output).
- Runtime dependencies unchanged: numpy, onnxruntime, soundfile, scipy.
- Model output: no change to head weights or runtime code; ONNX built from the same heads (held-out R4 gate PASS, see r4/result.md).

No differing API output, so no decision card needed. Patch bump is valid under semver.
