#!/usr/bin/env python3
"""Draw the R4 release-gate set: n=1200 Common Voice 17 English clips from the speakers reserved in R2 prep.

Reserved speakers: h = int(sha256('agr2:' + client_id)[:8], 16) % 10 == 1. R2 trained on h >= 2 and screened on h == 0
(research/microbench/age-gender-audio-r2/prep.py in the lab repo), so these speakers were never trained on, never
used for model selection, never scored. This script uses ONLY non-audio metadata columns to choose; no model is run.

Rule (deterministic, SEED fixed here before any model number exists):
  pool    = en rows of all shards given, both age bucket and gender mappable (non-binary/other skipped), h == 1
  per speaker: seeded shuffle of that speaker's clips; keep the first K (K = smallest in 1..MAX_K whose cell-capped pool reaches N)
  cells   = (age bucket, gender); cap C per cell = smallest value whose capped total reaches N (clips taken in seeded order)
  final   = seeded random drop down to exactly N, sorted by (shard, row)
Assertions printed and enforced: zero client_id overlap with the R2 train list and screen list, every drawn speaker has h == 1.
Output: OUT/r4_clips.json (list rows incl. client_id and audio sha256), OUT/r4_clips.sha256, OUT/r4_draw_report.json,
and (optional) AUDIO_DIR/NNNN.mp3 (the clip bytes, row order = r4_clips.json).
Usage: draw_r4.py RAW_DIR R2_PREP_DIR OUT_DIR [AUDIO_DIR] [--n N]   (N defaults to 1200; other values are for tests only)
"""
import argparse, collections, hashlib, json, os, random, sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parquet_clips import load_blobs

_ap = argparse.ArgumentParser()
_ap.add_argument("raw"); _ap.add_argument("prep"); _ap.add_argument("out"); _ap.add_argument("audio_dir", nargs="?")
_ap.add_argument("--n", type=int, default=1200)
_a = _ap.parse_args()
RAW, PREP, OUT, AUDIO_DIR = _a.raw, _a.prep, _a.out, _a.audio_dir
SEED, N, MAX_K = 20261124, _a.n, 5
BUCKET_MID = {"teens": 15, "twenties": 25, "thirties": 35, "fourties": 45, "fifties": 55,
              "sixties": 65, "seventies": 75, "eighties": 85, "nineties": 95}
GENDER = {"female_feminine": "female", "male_masculine": "male"}


def spk_hash(cid):
    return int(hashlib.sha256(("agr2:" + cid).encode()).hexdigest()[:8], 16) % 10


shards = sorted(f[:-len(".parquet")] for f in os.listdir(RAW) if f.startswith("en-train-") and f.endswith(".parquet"))
rows, n_total = [], 0
for shard in shards:
    t = pq.read_table(os.path.join(RAW, shard + ".parquet"), columns=["client_id", "path", "age", "gender"])
    cid, path, age, gen = (t.column(c).to_pylist() for c in ("client_id", "path", "age", "gender"))
    unk_a = {a for a in age if a and a not in BUCKET_MID}
    unk_g = {g for g in gen if g and g not in GENDER and g not in ("other", "non-binary")}
    if unk_a or unk_g:
        raise SystemExit(f"unmapped labels in {shard}: age={unk_a} gender={unk_g}")
    n_total += t.num_rows
    for i in range(t.num_rows):
        if age[i] in BUCKET_MID and gen[i] in GENDER and spk_hash(cid[i]) == 1:
            rows.append({"shard": shard, "row": i, "client_id": cid[i], "path": str(path[i]).split("/")[-1],
                         "age": BUCKET_MID[age[i]], "gender": GENDER[gen[i]]})
print(f"shards {len(shards)} rows {n_total}; h==1 labelled pool: clips {len(rows)} speakers {len({r['client_id'] for r in rows})}")

by_spk = collections.defaultdict(list)
for r in rows:
    by_spk[r["client_id"]].append(r)
rng = random.Random(SEED)
spk_order = sorted(by_spk)
for c in spk_order:
    by_spk[c].sort(key=lambda r: (r["shard"], r["row"]))   # canonical order before the seeded shuffle
    rng.shuffle(by_spk[c])
# a speaker has one gender/age label in practice; keep per-clip labels but report any speaker with mixed labels
mixed = sum(1 for c in spk_order if len({(r["age"], r["gender"]) for r in by_spk[c]}) > 1)
print("speakers with mixed (age, gender) labels across their clips:", mixed)


def capped_total(pool, C):
    cnt = collections.Counter((r["age"], r["gender"]) for r in pool)
    return sum(min(n, C) for n in cnt.values())


K = None
for k in range(1, MAX_K + 1):
    pool_k = [r for c in spk_order for r in by_spk[c][:k]]
    if len(pool_k) >= N:
        K = k
        break
assert K, "pool smaller than N even at MAX_K clips per speaker"
pool = [r for c in spk_order for r in by_spk[c][:K]]
rng.shuffle(pool)
cell_n = collections.Counter((r["age"], r["gender"]) for r in pool)
C = next(c for c in range(1, 100000) if capped_total(pool, c) >= N)
cnt, picked = collections.Counter(), []
for r in pool:
    k_ = (r["age"], r["gender"])
    if cnt[k_] < C:
        cnt[k_] += 1
        picked.append(r)
if len(picked) > N:
    drop = set(random.Random(SEED + 1).sample(range(len(picked)), len(picked) - N))
    picked = [r for i, r in enumerate(picked) if i not in drop]
picked.sort(key=lambda r: (r["shard"], r["row"]))
print(f"clips per speaker K={K}, cell cap C={C}, capped pool {sum(min(n, C) for n in cell_n.values())}, final {len(picked)}")
assert len(picked) == N

# ---- split assertions against everything that trained, selected or screened a model in R2
train_spk = {r["client_id"] for r in json.load(open(os.path.join(PREP, "train.json")))}
screen_spk = {r["client_id"] for r in json.load(open(os.path.join(PREP, "screen600.json")))}
r4_spk = {r["client_id"] for r in picked}
ov_tr, ov_sc = r4_spk & train_spk, r4_spk & screen_spk
print(f"SPLIT ASSERT: R4 speakers {len(r4_spk)} (clips {len(picked)}), R2 train-list speakers {len(train_spk)}, "
      f"R2 screen speakers {len(screen_spk)}; client_id overlap with train = {len(ov_tr)}, with screen = {len(ov_sc)}")
assert not ov_tr and not ov_sc
assert all(spk_hash(r["client_id"]) == 1 for r in picked)

# ---- pin the audio bytes: sha256 per clip goes INTO the hashed list
blobs = load_blobs(picked, RAW)
for r, b in zip(picked, blobs):
    assert b, r
    r["audio_sha256"] = hashlib.sha256(b).hexdigest()
if AUDIO_DIR:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    for i, b in enumerate(blobs):
        open(os.path.join(AUDIO_DIR, f"{i:04d}.mp3"), "wb").write(b)

os.makedirs(OUT, exist_ok=True)
fn = os.path.join(OUT, "r4_clips.json")
with open(fn, "w") as f:
    json.dump(picked, f, indent=0, sort_keys=True)
sha = hashlib.sha256(open(fn, "rb").read()).hexdigest()
open(os.path.join(OUT, "r4_clips.sha256"), "w").write(f"{sha}  r4_clips.json\n")
spk_counts = collections.Counter(r["client_id"] for r in picked)
rep = {"seed": SEED, "n": len(picked), "speakers": len(r4_spk), "max_clips_per_speaker": max(spk_counts.values()),
       "clips_per_speaker_K": K, "cell_cap_C": C, "pool_clips": len(rows), "pool_speakers": len(by_spk),
       "shards": shards[0] + ".." + shards[-1], "n_shards": len(shards), "overlap_with_train": len(ov_tr),
       "overlap_with_screen": len(ov_sc), "sha256_r4_clips_json": sha,
       "by_bucket": dict(sorted(collections.Counter(r["age"] for r in picked).items())),
       "by_gender": dict(collections.Counter(r["gender"] for r in picked)),
       "hash_rule": "int(sha256('agr2:'+client_id)[:8],16)%10 == 1"}
json.dump(rep, open(os.path.join(OUT, "r4_draw_report.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
print("R4 LIST SHA256", sha)
