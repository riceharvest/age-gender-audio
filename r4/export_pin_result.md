# Export pin: transformers >=4.57,<5

- Install extra: `export = ["torch", "transformers>=4.57,<5", "onnx"]`; `export_fp32` calls
  `assert_transformers_version()` and fails with `export needs transformers >=4.57,<5, found 5.24.0`.
- Clean venv install of the extra resolves: transformers 4.57.6, torch 2.15.0+cpu, onnx 1.23.2,
  onnxruntime 1.32.0.
- `tests/test_export_asserts.py`: 7 passed (stub 5.24.0 gives the one-line message); with the guard
  disabled both guard tests fail (DID NOT RAISE). Full `pytest tests/`: 16 passed, 2 skipped.
- Re-export at the pin (31 s, run twice, identical): fp32 sha256
  `3a28b86f2c2ab9b7f8d2fe4119722f398b7f6caf76fd8aa420c5a85d372d1bc4`; the earlier documented
  `d9d7d004...` came from transformers 5.5.4 and is not reproducible with the pinned range. int8
  sha256 `71436e54...65df` unchanged.
- 0.1.0 predates this pin and its wheel installs an unbounded `transformers` (5.x), which fails the
  wav2vec2 export with `IndexError: Dimension out of range`; 0.1.1 carries the pin.
