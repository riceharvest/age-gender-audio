#!/usr/bin/env python3
"""R4 release gate (pre-registered, dive s8; frozen with the code). Pure numpy, no model is run here.

Inputs: the frozen clip list (r4_clips.json: age = decade midpoint, gender), the incumbent re-scored on the SAME clips
(R2 rescore.py output: age_pred_10s, gender_probs_10s [female, male, child]) and our ONNX predictions
(score_onnx.py output: age_fp32/age_int8, logits_fp32/logits_int8 [female, male, child]).
Scoring: gender = argmax over [female, male] only (child ignored for both arms), accuracy; age = MAE vs the decade midpoint.
Paired 95% bootstrap over clips (one clip per client_id in r4_clips.json, so clip == speaker resampling), B=2000, fixed seed.
Gate (ours = int8 ONNX, the shipped artifact; fp32 is reported alongside):
  gender: lower CI bound of (ours - incumbent) accuracy >= -2.0 pp      age: upper CI bound of (ours - incumbent) MAE <= +1.0 yr
  parity: |acc_int8 - acc_fp32| <= 0.5 pp and |MAE_int8 - MAE_fp32| <= 0.3 yr on the same clips
Usage: gate_r4.py LIST.json INCUMBENT.json OURS.json OUT.json [--arm int8] [--gender-margin 2.0] [--age-margin 1.0]
       [--seed 20261124] [--age-key K --logits-key K] (the last two read R2-style prediction files, for pipeline validation)
"""
import argparse, hashlib, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("lst"); ap.add_argument("inc"); ap.add_argument("ours"); ap.add_argument("out")
ap.add_argument("--arm", default="int8"); ap.add_argument("--inc-key", default="10s")
ap.add_argument("--gender-margin", type=float, default=2.0); ap.add_argument("--age-margin", type=float, default=1.0)
ap.add_argument("--parity-gender-pp", type=float, default=0.5); ap.add_argument("--parity-age-yr", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=20261124); ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--age-key"); ap.add_argument("--logits-key")
a = ap.parse_args()

lst, inc, ours = (json.load(open(f)) for f in (a.lst, a.inc, a.ours))
n = len(lst)
assert n == len(inc) == len(ours), (n, len(inc), len(ours))
for x, y, z in zip(lst, inc, ours):
    assert (x["shard"], x["row"]) == (y["shard"], y["row"]) == (z["shard"], z["row"]), (x, y, z)
print("list sha256", hashlib.sha256(open(a.lst, "rb").read()).hexdigest(), "n", n)
age = np.array([r["age"] for r in lst], float)
male = np.array([r["gender"] == "male" for r in lst])


def arm(rows, age_key, lg_key, probs=False):
    ag = np.array([r[age_key] for r in rows], float)
    lg = np.array([r[lg_key] for r in rows], float)
    gc = ((lg[:, 1] > lg[:, 0]) == male).astype(float)          # 2-way argmax, child ignored
    return gc, np.abs(ag - age), int((lg.argmax(1) == 2).sum())


inc_gc, inc_ae, inc_child = arm(inc, f"age_pred_{a.inc_key}", f"gender_probs_{a.inc_key}")
if a.age_key:
    arms = {a.arm: arm(ours, a.age_key, a.logits_key)}
else:
    arms = {k: arm(ours, f"age_{k}", f"logits_{k}") for k in ("fp32", "int8")}
rng = np.random.RandomState(a.seed)
idx = rng.randint(0, n, (a.boot, n))
res = {"n": n, "seed": a.seed, "boot": a.boot, "arm": a.arm, "list_sha256": hashlib.sha256(open(a.lst, "rb").read()).hexdigest(),
       "incumbent": {"gender_acc_2way": float(inc_gc.mean()), "age_mae": float(inc_ae.mean()), "top_class_child": inc_child},
       "controls": {"majority_gender_acc": float(max(male.mean(), 1 - male.mean())), "mean_age_mae": float(np.abs(age - age.mean()).mean())},
       "arms": {}}
print(f"incumbent: gender acc {inc_gc.mean():.4f}, age MAE {inc_ae.mean():.3f}, top-class child {inc_child}/{n}")
for k, (gc, ae, ch) in arms.items():
    dg, da = (gc - inc_gc) * 100, ae - inc_ae
    gci, aci = np.percentile(dg[idx].mean(1), [2.5, 97.5]), np.percentile(da[idx].mean(1), [2.5, 97.5])
    g_ok, a_ok = bool(gci[0] >= -a.gender_margin), bool(aci[1] <= a.age_margin)
    res["arms"][k] = {"gender_acc": float(gc.mean()), "age_mae": float(ae.mean()), "child_top_class": ch,
                      "gender_diff_pp": float(dg.mean()), "gender_ci": gci.tolist(), "age_diff_yr": float(da.mean()), "age_ci": aci.tolist(),
                      "gender_ok": g_ok, "age_ok": a_ok, "discordant_pct": float((gc != inc_gc).mean() * 100)}
    print(f"[{k}] ours gender acc {gc.mean():.4f} age MAE {ae.mean():.3f}; diff gender {dg.mean():+.2f} pp CI [{gci[0]:+.2f},{gci[1]:+.2f}] "
          f"(need lower >= -{a.gender_margin}: {g_ok}); diff age {da.mean():+.3f} yr CI [{aci[0]:+.3f},{aci[1]:+.3f}] (need upper <= +{a.age_margin}: {a_ok})")
primary = res["arms"][a.arm]
res["gate_ci_pass"] = primary["gender_ok"] and primary["age_ok"]
if "fp32" in arms and "int8" in arms:
    dpp = (arms["int8"][0].mean() - arms["fp32"][0].mean()) * 100
    dyr = arms["int8"][1].mean() - arms["fp32"][1].mean()
    res["parity"] = {"gender_acc_delta_pp": float(dpp), "age_mae_delta_yr": float(dyr),
                     "ok": bool(abs(dpp) <= a.parity_gender_pp and abs(dyr) <= a.parity_age_yr)}
    print(f"parity int8-fp32: gender {dpp:+.2f} pp (limit {a.parity_gender_pp}), age MAE {dyr:+.3f} yr (limit {a.parity_age_yr}): {res['parity']['ok']}")
    res["gate_pass"] = bool(res["gate_ci_pass"] and res["parity"]["ok"])
else:
    res["gate_pass"] = None
print("GATE CI:", res["gate_ci_pass"], " GATE (CI + parity):", res["gate_pass"])
json.dump(res, open(a.out, "w"), indent=1)
