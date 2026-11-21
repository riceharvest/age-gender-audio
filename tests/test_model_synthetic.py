import io
import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
sf = pytest.importorskip("soundfile")
from onnx import TensorProto, helper, numpy_helper

from age_gender_audio import cli
from age_gender_audio.model import AgeGenderModel


def _tiny_model(path):
    """audio [1,T] -> embedding [1,768] (mean of the audio tiled), age [1,1], gender_logits [1,3] (female, male, child)."""
    wa = np.zeros((768, 1), np.float32); wa[:, 0] = 0.01
    wg = np.zeros((768, 3), np.float32); wg[:, 0] = -1.0; wg[:, 1] = 1.0; wg[:, 2] = 5.0  # child logit largest: must be ignored
    inits = [numpy_helper.from_array(np.ones((1, 768), np.float32), "ones"), numpy_helper.from_array(wa, "wa"),
             numpy_helper.from_array(wg, "wg")]
    nodes = [helper.make_node("ReduceMean", ["audio"], ["m"], axes=[1], keepdims=1),
             helper.make_node("MatMul", ["m", "ones"], ["embedding"]),
             helper.make_node("MatMul", ["embedding", "wa"], ["age"]),
             helper.make_node("MatMul", ["embedding", "wg"], ["gender_logits"])]
    g = helper.make_graph(nodes, "t", [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "T"])],
                          [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 768]),
                           helper.make_tensor_value_info("age", TensorProto.FLOAT, [1, 1]),
                           helper.make_tensor_value_info("gender_logits", TensorProto.FLOAT, [1, 3])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10  # newer onnx writes IR 14; older onnxruntime (py3.10 CI) loads up to IR 11
    onnx.save(m, path)


def _wav(path):
    x = (0.1 * np.sin(np.linspace(0, 400, 16000))).astype(np.float32) + 0.05
    sf.write(path, x, 16000)


def test_forward_and_predict(tmp_path):
    _tiny_model(str(tmp_path / "m.onnx")); _wav(str(tmp_path / "a.wav"))
    m = AgeGenderModel(str(tmp_path / "m.onnx"))
    emb, age, probs = m.forward(np.sin(np.linspace(0, 50, 16000)).astype(np.float32), 16000)
    assert emb.shape == (768,) and age.shape == (1,) and probs.shape == (3,)
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    r = m.predict(str(tmp_path / "a.wav"))
    assert set(r) == {"age_years", "gender", "gender_probs"}
    assert r["gender"] in ("female", "male")  # child is never returned


def test_too_short_audio_rejected(tmp_path):
    _tiny_model(str(tmp_path / "m.onnx"))
    with pytest.raises(ValueError):
        AgeGenderModel(str(tmp_path / "m.onnx")).run_raw(np.zeros(100, np.float32))


def test_cli_json_line_and_missing_model(tmp_path, capsys, monkeypatch):
    _tiny_model(str(tmp_path / "m.onnx")); _wav(str(tmp_path / "a.wav"))
    monkeypatch.delenv(cli.ENV, raising=False)
    assert cli.main([str(tmp_path / "a.wav")]) == 2
    assert cli.main([str(tmp_path / "a.wav"), "--model", str(tmp_path / "m.onnx")]) == 0
    d = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert d["file"].endswith("a.wav") and "age_years" in d
