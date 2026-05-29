#!/usr/bin/env python3

"""Pack per-caption text feature files into one NPZ per Clotho-Moment qid.

Expected input file naming convention:
  qid{qid}_caption{index}.npz

Each input NPZ must contain a single array under the key `last_hidden_state`.
The packed output keeps the same qid filename and stores one key per caption:
  caption_1, caption_2, ..., caption_5

This format lets the dataset loader randomly sample one caption embedding at
training time without changing the rest of the pipeline.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


INPUT_PATTERN = re.compile(r"^qid(?P<qid>.+?)_caption(?P<caption_index>\d+)\.npz$")


def load_feature(path, key):
    with np.load(path) as data:
        if key not in data.files:
            raise KeyError(f"Missing key {key!r} in {path}")
        return data[key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing qid*_caption*.npz files")
    parser.add_argument("--output-dir", required=True, help="Directory to write packed qid*.npz files")
    parser.add_argument("--input-key", default="last_hidden_state", help="Feature key inside each input NPZ")
    parser.add_argument("--strict", action="store_true", help="Fail if a qid does not have five captions")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_paths = defaultdict(dict)
    for path in sorted(input_dir.glob("qid*_caption*.npz")):
        match = INPUT_PATTERN.match(path.name)
        if match is None:
            continue
        qid = match.group("qid")
        caption_index = int(match.group("caption_index"))
        grouped_paths[qid][caption_index] = path

    if not grouped_paths:
        raise FileNotFoundError(f"No qid*_caption*.npz files found under {input_dir}")

    packed_count = 0
    for qid, caption_map in sorted(grouped_paths.items()):
        if args.strict and len(caption_map) != 5:
            raise ValueError(f"qid{qid} has {len(caption_map)} captions, expected 5")

        packed = {}
        for caption_index in sorted(caption_map):
            feature = load_feature(caption_map[caption_index], args.input_key)
            packed[f"caption_{caption_index}"] = feature

        output_path = output_dir / f"qid{qid}.npz"
        np.savez_compressed(output_path, **packed)
        packed_count += 1

    print(f"Packed {packed_count} qid feature files into {output_dir}")


if __name__ == "__main__":
    main()
