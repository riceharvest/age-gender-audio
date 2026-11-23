"""age-gender-audio: age and gender from speech, wav2vec2-base heads, int8 ONNX, drop-in output shape."""

__version__ = "0.1.1"


def __getattr__(name):  # lazy: importing the package must not import onnxruntime
    if name == "AgeGenderModel":
        from .model import AgeGenderModel

        return AgeGenderModel
    raise AttributeError(name)
