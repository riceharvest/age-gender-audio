import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from age_gender_audio import export


def _graph(nodes):
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    p = None
    return helper.make_model(g)


def _mmi(i, name=None):
    parts = ["attention/q_proj", "attention/k_proj", "attention/v_proj", "attention/out_proj",
             "feed_forward/intermediate_dense", "feed_forward/output_dense"]
    l, k = divmod(i, 6)
    return helper.make_node("MatMulInteger", ["a", "b"], [f"o{i}"], name=name or f"/layers.{l}/{parts[k]}/MatMul_quant")


def _save(tmp_path, nodes):
    p = str(tmp_path / "m.onnx")
    onnx.save(_graph(nodes), p)
    return p


def test_passes_on_exactly_the_72_layer_matmuls(tmp_path):
    p = _save(tmp_path, [_mmi(i) for i in range(72)] + [helper.make_node("Conv", ["x", "w"], ["c"], name="conv0")])
    info = export.assert_int8_graph(p)
    assert info["matmul_integer"] == 72 and info["conv"] == 1


def test_fails_on_silently_unquantized_export(tmp_path):
    p = _save(tmp_path, [helper.make_node("MatMul", ["a", "b"], ["o"], name="/layers.0/attention/q_proj/MatMul")])
    with pytest.raises(AssertionError, match="0 MatMulInteger"):
        export.assert_int8_graph(p)


def test_fails_when_front_end_is_quantized(tmp_path):
    nodes = [_mmi(i) for i in range(72)] + [helper.make_node("ConvInteger", ["a", "b"], ["c"], name="conv0")]
    with pytest.raises(AssertionError, match="front-end"):
        export.assert_int8_graph(_save(tmp_path, nodes))


def test_fails_on_wrong_count_or_stray_node(tmp_path):
    with pytest.raises(AssertionError, match="expected 72"):
        export.assert_int8_graph(_save(tmp_path, [_mmi(i) for i in range(71)]))
    nodes = [_mmi(i) for i in range(71)] + [_mmi(0, name="/trunk/Gemm_MatMul_quant")]
    with pytest.raises(AssertionError, match="outside"):
        export.assert_int8_graph(_save(tmp_path, nodes))


def test_layer_regex_skips_activation_matmuls_and_heads():
    r = export.LAYER_WEIGHT_MATMUL
    assert r.match("/layers.11/feed_forward/output_dense/MatMul")
    assert not r.match("/layers.0/attention/MatMul") and not r.match("/layers.0/attention/MatMul_1")
    assert not r.match("/trunk/Gemm")
