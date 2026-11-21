"""CLI entry point: age-gender-audio FILE [FILE ...] [--model PATH.onnx] -> one JSON line per file."""
import argparse
import json
import os
import sys

from . import __version__

ENV = "AGE_GENDER_AUDIO_ONNX"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="age-gender-audio", description="Estimate age (decade-bucket resolution) and gender from speech.")
    p.add_argument("files", nargs="+")
    p.add_argument("--model", default=os.environ.get(ENV), help=f"ONNX file (or set {ENV})")
    p.add_argument("--version", action="version", version=__version__)
    a = p.parse_args(argv)
    if not a.model:
        print(f"error: no model file: pass --model PATH.onnx or set {ENV}", file=sys.stderr)
        return 2
    from .model import AgeGenderModel

    m = AgeGenderModel(a.model)
    for f in a.files:
        print(json.dumps({"file": f, **m.predict(f)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
