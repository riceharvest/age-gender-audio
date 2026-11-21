"""Build the ONNX files: fp32 graph (frozen wav2vec2-base + trained heads) and the int8 transformer-only graph.

Needs the `export` extra: torch, transformers (>=4.57,<5, asserted in export_fp32), onnx, onnxruntime. The backbone is fetched from the Hub by id
(facebook/wav2vec2-base, Apache-2.0); nothing from the incumbent is used.

Int8 rule (lab METHODOLOGY 3b.9, dive s8): wav2vec2 collapses if the CNN front end is quantized, so ONLY the
MatMul weights inside the 12 transformer layers (q,k,v,out,fc1 = 60 nodes) are quantized; the conv feature
encoder, positional conv, feature projection, the second feed-forward matmul (fc2) and the heads stay fp32.
Why fc2 stays fp32: its input is a GELU output with outliers; quantizing it alone moved the age prediction by 2.5 yr
(mean |diff| vs fp32, 40 clips of the already-spent R2 screen set, per-channel reduce-range) against <= 1.0 yr for each
of the other five; all six together gave 3.1 yr (uint8 per-channel). The recipe was chosen on that spent set, never on
the R4 clips.
`assert_int8_graph` checks the result and fails loudly on a silently-unquantized export (0 MatMulInteger) or any ConvInteger / extra quantized node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re


from . import heads

BACKBONE = "facebook/wav2vec2-base"
N_LAYERS = 12
QUANT_PER_LAYER = 5  # q_proj, k_proj, v_proj, out_proj, fc1 (fc2 stays fp32, see module doc)
EXPECTED_MATMULINTEGER = N_LAYERS * QUANT_PER_LAYER
OPSET = 17
TRANSFORMERS_RANGE = ">=4.57,<5"  # same bound as the `export` extra in pyproject.toml; 5.x cannot export wav2vec2 (attention IndexError)


def _parse_ver(v: str) -> tuple[int, int]:
    m = re.match(r"(\d+)\.(\d+)", v)
    if not m:
        raise AssertionError(f"export needs transformers {TRANSFORMERS_RANGE}, found unparseable version {v!r}")
    return int(m.group(1)), int(m.group(2))


def assert_transformers_version(found: str | None = None) -> str:
    """Fail fast (one line) unless the installed transformers is inside TRANSFORMERS_RANGE."""
    if found is None:
        import transformers
        found = transformers.__version__
    if not ((4, 57) <= _parse_ver(found) < (5, 0)):
        raise AssertionError(f"export needs transformers {TRANSFORMERS_RANGE}, found {found}")
    return found


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

    assert_transformers_version()
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
    r"^/layers\.\d+/(attention/(q_proj|k_proj|v_proj|out_proj)|feed_forward/intermediate_dense)/MatMul$")


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


def quantize_transformer_int8(fp32_path: str, int8_path: str) -> dict:
    """Dynamic int8: uint8 weights (per channel; avoids the AVX2 u8s8 saturation), uint8 activations computed at runtime."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    m = onnx.load(fp32_path)
    keep, excl = _split_nodes(m)
    if len(keep) != EXPECTED_MATMULINTEGER:
        raise AssertionError(f"{fp32_path}: {len(keep)} layer weight MatMuls found, expected {EXPECTED_MATMULINTEGER}")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Gemm"],
                     nodes_to_exclude=excl, per_channel=True, reduce_range=False)
    return assert_int8_graph(int8_path)


def assert_int8_graph(path: str) -> dict:
    """Fail unless exactly the 60 transformer-layer MatMuls are int8 and nothing else is quantized."""
    import onnx

    m = onnx.load(path)
    ops: dict[str, int] = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    n_mmi = ops.get("MatMulInteger", 0)
    if n_mmi == 0:
        raise AssertionError(f"{path}: 0 MatMulInteger nodes: the export is silently unquantized")
    if n_mmi != EXPECTED_MATMULINTEGER:
        raise AssertionError(f"{path}: {n_mmi} MatMulInteger nodes, expected {EXPECTED_MATMULINTEGER} (12 layers x 5)")
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
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    fp32, int8 = os.path.join(a.outdir, "age_gender_fp32.onnx"), os.path.join(a.outdir, "age_gender_int8.onnx")
    export_fp32(fp32)
    info = quantize_transformer_int8(fp32, int8)
    rep = {"fp32": {"path": fp32, "bytes": os.path.getsize(fp32), "sha256": sha256(fp32)},
           "int8": {"path": int8, "bytes": os.path.getsize(int8), "sha256": sha256(int8), **info},
           "backbone": BACKBONE, "opset": OPSET}
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
