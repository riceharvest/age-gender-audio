"""Build the ONNX files: fp32 graph (frozen wav2vec2-base + trained heads) and the int8 transformer-only graph.

Needs the `export` extra: torch, transformers, onnx, onnxruntime. The backbone is fetched from the Hub by id
(facebook/wav2vec2-base, Apache-2.0); nothing from the incumbent is used.

Int8 rule (lab METHODOLOGY 3b.9, dive s8): wav2vec2 collapses if the CNN front end is quantized, so ONLY the
MatMul weights inside the 12 transformer layers (q,k,v,out,fc1,fc2 = 72 nodes) are quantized; the conv feature
encoder, positional conv, feature projection and the heads stay fp32. `assert_int8_graph` checks the result
and fails loudly on a silently-unquantized export (0 MatMulInteger) or any ConvInteger / extra quantized node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

import numpy as np

from . import heads

BACKBONE = "facebook/wav2vec2-base"
N_LAYERS = 12
QUANT_PER_LAYER = 6  # q_proj, k_proj, v_proj, out_proj, fc1, fc2
EXPECTED_MATMULINTEGER = N_LAYERS * QUANT_PER_LAYER
OPSET = 17


def build_torch_module(head_path: str = heads.HEAD_FILE, backbone: str = BACKBONE):
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model

    h = heads.load_head(head_path)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Wav2Vec2Model.from_pretrained(backbone, attn_implementation="eager").eval()
            self.register_buffer("mu", torch.from_numpy(h["mu"]))
            self.register_buffer("sd", torch.from_numpy(h["sd"]))
            self.trunk = nn.Linear(768, 256)
            self.age = nn.Linear(256, 1)
            self.gen = nn.Linear(256, 3)
            with torch.no_grad():
                self.trunk.weight.copy_(torch.from_numpy(h["w1"])); self.trunk.bias.copy_(torch.from_numpy(h["b1"]))
                self.age.weight.copy_(torch.from_numpy(h["wa"])); self.age.bias.copy_(torch.from_numpy(h["ba"]))
                self.gen.weight.copy_(torch.from_numpy(h["wg"])); self.gen.bias.copy_(torch.from_numpy(h["bg"]))
            for p in self.parameters():
                p.requires_grad_(False)

        def encode(self, audio):
            """Same computation as Wav2Vec2Model.forward with attention_mask=None, written out because the HF mask
            helper (transformers 5) cannot be traced for ONNX export. Checked against the HF forward in export_fp32."""
            b = self.backbone
            hs, _ = b.feature_projection(b.feature_extractor(audio).transpose(1, 2))
            enc = b.encoder
            hs = enc.layer_norm(hs + enc.pos_conv_embed(hs))
            for layer in enc.layers:
                hs = layer(hs, attention_mask=None)[0]
            return hs

        def forward(self, audio):
            emb = self.encode(audio).mean(1)  # == hidden_states[12] mean-pooled (R2 layer 12)
            z = torch.nn.functional.gelu(self.trunk((emb - self.mu) / self.sd))
            return emb, self.age(z), self.gen(z)

    return Net().eval()


def export_fp32(out_path: str, head_path: str = heads.HEAD_FILE, backbone: str = BACKBONE) -> str:
    import torch

    net = build_torch_module(head_path, backbone)
    x = torch.randn(1, 32000)
    with torch.no_grad():
        ref = net.backbone(x).last_hidden_state
        diff = (net.encode(x) - ref).abs().max().item()
    assert diff < 1e-4, f"manual encoder forward differs from the HF forward by {diff}"
    torch.onnx.export(net, (x,), out_path, input_names=["audio"], output_names=["embedding", "age", "gender_logits"],
                      dynamic_axes={"audio": {1: "samples"}}, opset_version=OPSET, dynamo=False)
    return out_path


# weight MatMuls of the 12 transformer layers (node names as written by torch.onnx for this module); the two
# activation x activation attention matmuls (QK^T, AV) are deliberately not matched.
LAYER_WEIGHT_MATMUL = re.compile(
    r"^/layers\.\d+/(attention/(q_proj|k_proj|v_proj|out_proj)|feed_forward/(intermediate_dense|output_dense))/MatMul$")


def _split_nodes(model) -> tuple[list[str], list[str]]:
    """(names to quantize, names to exclude): every MatMul/Gemm node is either a layer weight MatMul or excluded."""
    keep, excl = [], []
    for n in model.graph.node:
        if n.op_type in ("MatMul", "Gemm"):
            if LAYER_WEIGHT_MATMUL.match(n.name):
                keep.append(n.name)
            else:
                excl.append(n.name)
                if n.op_type == "Gemm":  # ORT rewrites Gemm -> MatMul+Add named "<name>_MatMul" before it applies the exclusions
                    excl.append(n.name + "_MatMul")
    return keep, excl


# ---- SmoothQuant-style activation smoothing (exact in fp32; moves activation outliers into the weights)
SMOOTH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "smooth.npz")
GROUPS = {"qkv": ("attention/q_proj", "attention/k_proj", "attention/v_proj"), "out": ("attention/out_proj",),
          "fc1": ("feed_forward/intermediate_dense",), "fc2": ("feed_forward/output_dense",)}


def _group_nodes(model):
    """{(layer, group): [MatMul nodes]} for the 72 layer weight MatMuls."""
    by_name = {n.name: n for n in model.graph.node}
    out = {}
    for l in range(N_LAYERS):
        for g, parts in GROUPS.items():
            out[(l, g)] = [by_name[f"/layers.{l}/{p}/MatMul"] for p in parts]
    return out


def collect_act_stats(fp32_path: str, clips, threads: int = 4) -> dict:
    """Run the fp32 graph on preprocessed clips; per (layer, group): per-channel max |activation| entering the MatMul(s)."""
    import onnx
    import onnxruntime as ort

    m = onnx.load(fp32_path)
    names = {k: nodes[0].input[0] for k, nodes in _group_nodes(m).items()}
    for k, nodes in _group_nodes(m).items():
        assert len({n.input[0] for n in nodes}) == 1, k
    for t in sorted(set(names.values())):
        m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    order = sorted(set(names.values()))
    stats = {k: None for k in names}
    for x in clips:
        outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"audio": np.asarray(x, np.float32)[None]})))
        for k, t in names.items():
            v = np.abs(outs[t]).reshape(-1, outs[t].shape[-1]).max(0)
            stats[k] = v if stats[k] is None else np.maximum(stats[k], v)
    return stats


def smooth_scales(fp32_path: str, stats: dict, alpha: float = 0.5) -> dict:
    """s_j = max|X_j|^alpha / max|W_j|^(1-alpha) per input channel j (W_j: max over the group's weight rows)."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(fp32_path)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    out = {}
    for (l, g), nodes in _group_nodes(m).items():
        w = np.max([np.abs(init[n.input[1]]).max(1) for n in nodes], axis=0)  # W is [in, out]
        x = np.maximum(stats[(l, g)], 1e-5)
        s = np.clip(x ** alpha / np.maximum(w, 1e-5) ** (1 - alpha), 1e-3, 1e3)
        out[f"{l}.{g}"] = s.astype(np.float32)
    return out


def apply_smoothing(fp32_path: str, out_path: str, scales: dict) -> None:
    """x -> x / s before each layer weight MatMul and W -> diag(s) W: the same function in fp32, easier to quantize."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(fp32_path)
    init = {i.name: i for i in m.graph.initializer}
    for (l, g), nodes in _group_nodes(m).items():
        s = np.asarray(scales[f"{l}.{g}"], np.float32)
        act = nodes[0].input[0]
        new = act + f"_smooth{l}{g}"
        cname = f"smooth_inv_{l}_{g}"
        m.graph.initializer.append(numpy_helper.from_array((1.0 / s).astype(np.float32), cname))
        first = min(m.graph.node.index(n) for n in nodes)
        m.graph.node.insert(first, onnx.helper.make_node("Mul", [act, cname], [new], name=f"/layers.{l}/smooth_{g}"))
        for n in nodes:
            n.input[0] = new
            w = numpy_helper.to_array(init[n.input[1]])
            init[n.input[1]].CopyFrom(numpy_helper.from_array((w * s[:, None]).astype(np.float32), n.input[1]))
    onnx.checker.check_model(m)
    onnx.save(m, out_path)


def quantize_transformer_int8(fp32_path: str, int8_path: str, reduce_range: bool = False, per_channel: bool = True,
                              smooth: dict | None = None, uint8_weights: bool = True) -> dict:
    import tempfile

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if smooth is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False, dir=os.path.dirname(int8_path) or ".")
        tmp.close()
        apply_smoothing(fp32_path, tmp.name, smooth)
        fp32_path = tmp.name
    m = onnx.load(fp32_path)
    keep, excl = _split_nodes(m)
    if len(keep) != EXPECTED_MATMULINTEGER:
        raise AssertionError(f"{fp32_path}: {len(keep)} layer weight MatMuls found, expected {EXPECTED_MATMULINTEGER}")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8 if uint8_weights else QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excl, per_channel=per_channel,
                     reduce_range=reduce_range)
    if smooth is not None:
        os.unlink(fp32_path)
    return assert_int8_graph(int8_path)


def assert_int8_graph(path: str) -> dict:
    """Fail unless exactly the 72 transformer-layer MatMuls are int8 and nothing else is quantized."""
    import onnx

    m = onnx.load(path)
    ops: dict[str, int] = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    n_mmi = ops.get("MatMulInteger", 0)
    if n_mmi == 0:
        raise AssertionError(f"{path}: 0 MatMulInteger nodes: the export is silently unquantized")
    if n_mmi != EXPECTED_MATMULINTEGER:
        raise AssertionError(f"{path}: {n_mmi} MatMulInteger nodes, expected {EXPECTED_MATMULINTEGER} (12 layers x 6)")
    bad = {k: v for k, v in ops.items() if k in ("ConvInteger", "QLinearConv", "QLinearMatMul")}
    if bad:
        raise AssertionError(f"{path}: quantized front-end ops present: {bad}")
    outside = [n.name for n in m.graph.node
               if n.op_type == "MatMulInteger" and not LAYER_WEIGHT_MATMUL.match(n.name.replace("_quant", ""))]
    if outside:
        raise AssertionError(f"{path}: MatMulInteger outside the transformer layers: {outside[:5]}")
    return {"ops": ops, "matmul_integer": n_mmi, "conv": ops.get("Conv", 0), "float_matmul": ops.get("MatMul", 0)}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="export fp32 and int8 (transformer-only) ONNX graphs")
    ap.add_argument("outdir")
    ap.add_argument("--reduce-range", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    fp32, int8 = os.path.join(a.outdir, "age_gender_fp32.onnx"), os.path.join(a.outdir, "age_gender_int8.onnx")
    export_fp32(fp32)
    info = quantize_transformer_int8(fp32, int8, a.reduce_range)
    rep = {"fp32": {"path": fp32, "bytes": os.path.getsize(fp32), "sha256": sha256(fp32)},
           "int8": {"path": int8, "bytes": os.path.getsize(int8), "sha256": sha256(int8), **info},
           "backbone": BACKBONE, "opset": OPSET, "reduce_range": a.reduce_range}
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
