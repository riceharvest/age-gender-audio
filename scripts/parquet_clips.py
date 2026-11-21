"""Read mp3 bytes of listed clips out of the raw CV17 parquet shards (same logic as the R2 extract.py)."""
import os
import pyarrow.parquet as pq


def load_blobs(items, raw_dir):
    """items: rows with 'shard' (e.g. 'en-train-0003') and 'row' -> list of audio bytes in the same order."""
    out = [None] * len(items)
    by_shard = {}
    for i, r in enumerate(items):
        by_shard.setdefault(r["shard"], []).append(i)
    for shard, idxs in by_shard.items():
        pf = pq.ParquetFile(os.path.join(raw_dir, shard + ".parquet"))
        offs, s = [], 0
        for g in range(pf.metadata.num_row_groups):
            offs.append(s)
            s += pf.metadata.row_group(g).num_rows
        need = {}
        for i in idxs:
            row = items[i]["row"]
            g = max(k for k, o in enumerate(offs) if o <= row)
            need.setdefault(g, []).append((i, row - offs[g]))
        for g, lst in need.items():
            col = pf.read_row_group(g, columns=["audio"]).column("audio")
            for i, off in lst:
                out[i] = col[off].as_py()["bytes"]
    return out
