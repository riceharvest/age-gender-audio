import hashlib
import json
import os
import subprocess
import sys

import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "draw_r4.py")
AGES = ["teens", "twenties", "thirties", "fourties", "fifties"]
GENS = ["female_feminine", "male_masculine"]


def _h(cid):
    return int(hashlib.sha256(("agr2:" + cid).encode()).hexdigest()[:8], 16) % 10


def _ids(h, n, tag):
    out, i = [], 0
    while len(out) < n:
        c = f"{tag}{i}"
        if _h(c) == h:
            out.append(c)
        i += 1
    return out


@pytest.fixture()
def world(tmp_path):
    raw, prep = tmp_path / "raw", tmp_path / "prep"
    raw.mkdir(); prep.mkdir()
    spk = {h: _ids(h, 30 if h == 1 else 6, f"s{h}_") for h in (0, 1, 2)}
    rows = []
    for h, ids in spk.items():
        for j, c in enumerate(ids):
            for k in range(2):  # two clips per speaker
                rows.append((c, f"{c}_{k}.mp3", AGES[j % 5], GENS[j % 2]))
    rows.append((spk[1][0], "nolabel.mp3", None, None))  # unlabeled h==1 row must be ignored
    rows.append((spk[1][1], "nb.mp3", "twenties", "non-binary"))  # skipped
    for s in (0, 1):
        part = rows[s::2]
        t = pa.table({"client_id": [r[0] for r in part], "path": [r[1] for r in part], "age": [r[2] for r in part],
                      "gender": [r[3] for r in part],
                      "audio": [{"bytes": r[1].encode() * 3, "path": r[1]} for r in part]})
        pq.write_table(t, str(raw / f"en-train-000{s}.parquet"), row_group_size=7)
    json.dump([{"client_id": c} for c in spk[2]], open(prep / "train.json", "w"))
    json.dump([{"client_id": c} for c in spk[0]], open(prep / "screen600.json", "w"))
    return raw, prep, spk


def _run(raw, prep, out, extra=()):
    return subprocess.run([sys.executable, SCRIPT, str(raw), str(prep), str(out), *extra, "--n", "40"],
                          capture_output=True, text=True)


def test_draw_is_deterministic_reserved_only_and_hashed(world, tmp_path):
    raw, prep, spk = world
    r1 = _run(raw, prep, tmp_path / "o1", [str(tmp_path / "audio")])
    r2 = _run(raw, prep, tmp_path / "o2")
    assert r1.returncode == 0, r1.stderr + r1.stdout
    a = json.load(open(tmp_path / "o1" / "r4_clips.json"))
    assert len(a) == 40
    assert {r["client_id"] for r in a} <= set(spk[1])
    assert all(_h(r["client_id"]) == 1 for r in a)
    assert all(r["path"] != "nolabel.mp3" and r["path"] != "nb.mp3" for r in a)
    sha = hashlib.sha256(open(tmp_path / "o1" / "r4_clips.json", "rb").read()).hexdigest()
    assert open(tmp_path / "o1" / "r4_clips.sha256").read().split()[0] == sha
    assert open(tmp_path / "o2" / "r4_clips.sha256").read() == open(tmp_path / "o1" / "r4_clips.sha256").read()
    rep = json.load(open(tmp_path / "o1" / "r4_draw_report.json"))
    assert rep["overlap_with_train"] == 0 and rep["overlap_with_screen"] == 0 and rep["n"] == 40
    for i, r in enumerate(a):  # audio bytes pinned in the hashed list
        assert hashlib.sha256(open(tmp_path / "audio" / f"{i:04d}.mp3", "rb").read()).hexdigest() == r["audio_sha256"]


def test_overlap_with_train_list_is_fatal(world, tmp_path):
    raw, prep, spk = world
    json.dump([{"client_id": c} for c in spk[1]], open(prep / "train.json", "w"))
    r = _run(raw, prep, tmp_path / "o")
    assert r.returncode != 0 and "overlap" in (r.stderr + r.stdout)
