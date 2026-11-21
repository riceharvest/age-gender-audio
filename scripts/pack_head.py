#!/usr/bin/env python3
"""Pack a trained R2 head checkpoint (train.py output: {'state','mu','sd','layer'}) into age_gender_audio/data/head.npz.

Usage: pack_head.py head_seed0.pt   (needs torch). Prints the sha256 of the source and of the packed file.
"""
import hashlib, os, sys
import numpy as np
import torch

src = sys.argv[1]
d = torch.load(src, map_location="cpu", weights_only=False)
assert d["layer"] == 12, d["layer"]
s = {k: v.numpy().astype(np.float32) for k, v in d["state"].items()}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "age_gender_audio", "data", "head.npz")
np.savez(out, mu=np.asarray(d["mu"], np.float32), sd=np.asarray(d["sd"], np.float32),
         w1=s["trunk.0.weight"], b1=s["trunk.0.bias"], wa=s["age.weight"], ba=s["age.bias"],
         wg=s["gen.weight"], bg=s["gen.bias"])
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print("source", src, sha(src)); print("packed", os.path.normpath(out), sha(out), os.path.getsize(out), "bytes")
