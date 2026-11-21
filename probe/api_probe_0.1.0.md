IDENTICAL (first release; baseline recorded). No 0.0.0 release exists: no git tag, not on PyPI (pip index shows none). The only earlier state is the scaffold commit eed1baf, whose placeholder `greet(name)` and `age-gender-audio [name]` CLI were never shipped or tagged. They are replaced, not an API break.

# Shipped surface baseline, age-gender-audio 0.1.0 (code f9d1cd1, source read at HEAD c19cc32)
- `age_gender_audio.__version__` = "0.1.0"
- `age_gender_audio.AgeGenderModel` (lazy import)
  - `__init__(onnx_path, providers=None, intra_op_threads=None)`
  - `run_raw(x[T] float32, >=400 samples) -> (embedding[768], age[1] 0..1, gender_logits[3])`; ONNX I/O names: in `audio`; out `embedding`, `age`, `gender_logits`
  - `forward(wav, sr=16000)` / `__call__` -> (embedding[768], age[1] 0..1, gender probs[3] female,male,child)
  - `predict(src) -> {"age_years": float, "gender": "female"|"male", "gender_probs": {"female","male","child"}}`
- CLI `age-gender-audio FILE [FILE ...] [--model PATH.onnx] [--version]`; env AGE_GENDER_AUDIO_ONNX; one JSON line per file `{"file", "age_years", "gender", "gender_probs"}`; exit 2 if no model.
- Deps: numpy, onnxruntime, soundfile, scipy.
