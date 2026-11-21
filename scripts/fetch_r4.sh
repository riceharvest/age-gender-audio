#!/bin/bash
# Re-fetch the CC0 Common Voice 17 English train parquet shards (fixie-ai mirror) the R4 draw reads: en/train/0000 .. 0089
# (90 shards, ~44 GB). Usage: fetch_r4.sh RAW_DIR [FIRST LAST]
set -e
RAW=${1:?RAW_DIR}; mkdir -p "$RAW"; cd "$RAW"
for i in $(seq -f "%04g" ${2:-0} ${3:-89}); do
  [ -s en-train-$i.parquet ] || { curl -sfL "https://huggingface.co/datasets/fixie-ai/common_voice_17_0/resolve/refs%2Fconvert%2Fparquet/en/train/$i.parquet" -o en-train-$i.parquet.tmp && mv en-train-$i.parquet.tmp en-train-$i.parquet; }
done
