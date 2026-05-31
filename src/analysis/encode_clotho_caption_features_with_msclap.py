#!/usr/bin/env python3

"""Encode Clotho-Moment caption manifest entries with msclap.

This reads the one-caption-per-line manifest produced by
`export_clotho_caption_manifest.py`, extracts token-level text features from
msclap, and writes one NPZ per caption:

  qid{qid}_caption{caption_index}.npz

Each output file contains a single `last_hidden_state` array with shape
`(T, 768)` and dtype `float32`.
"""

import argparse
import json
import logging
import os
import sys
import traceback
import tempfile
from pathlib import Path

import numpy as np
from msclap import CLAP
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def encode_caption(clap, caption):
    tokenized = clap.preprocess_text([caption])
    hidden_states = clap.clap.caption_encoder.base(**tokenized)[0]
    seq_len = int(tokenized["attention_mask"].sum().item())
    last_hidden_state = hidden_states[0, :seq_len, :].detach().cpu().numpy().astype(np.float32, copy=False)
    return last_hidden_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest path")
    parser.add_argument("--output-dir", required=True, help="Directory for qid*_caption*.npz files")
    parser.add_argument("--version", default="2023", help="msclap version to load")
    parser.add_argument("--use-cuda", action="store_true", help="Run msclap on CUDA")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of manifest rows to encode")
    parser.add_argument("--progress", action="store_true", help="Show progress bar (requires tqdm)")
    parser.add_argument("--print-every", type=int, default=1000, help="Print progress every N records")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clap = CLAP(version=args.version, use_cuda=args.use_cuda)

    total = 0
    written = 0

    # prepare progress display
    show_progress = args.progress and (tqdm is not None)
    estimated_total = None
    if show_progress:
        # try to count lines for progress bar
        try:
            with manifest_path.open("r", encoding="utf-8") as fh_count:
                estimated_total = sum(1 for _ in fh_count)
        except Exception:
            estimated_total = None

    with manifest_path.open("r", encoding="utf-8") as fh:
        iterator = fh
        if show_progress and estimated_total is not None:
            iterator = tqdm(fh, total=estimated_total, desc="encoding captions", unit="rec")

        for line in iterator:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception:
                logging.exception("Failed to parse manifest line: %r", line)
                continue

            total += 1
            if args.limit and total > args.limit:
                break

            qid = record.get("qid")
            caption_index = record.get("caption_index")
            caption = record.get("caption") or record.get("query")
            if qid is None or caption_index is None or not caption:
                continue

            try:
                last_hidden_state = encode_caption(clap, caption)
                output_path = output_dir / f"qid{qid}_caption{int(caption_index)}.npz"
                with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".npz.tmp", delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                try:
                    with tmp_path.open("wb") as fh:
                        np.savez_compressed(fh, last_hidden_state=last_hidden_state)
                    os.replace(tmp_path, output_path)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
                written += 1
            except Exception:
                logging.error("Encoding failed for qid=%s caption_index=%s", qid, caption_index)
                logging.error(traceback.format_exc())

            if not show_progress and (total % args.print_every == 0):
                print(f"Processed {total} lines, wrote {written} files", flush=True)

    print(f"Processed {total} manifest rows")
    print(f"Wrote {written} NPZ files to {output_dir}")


if __name__ == "__main__":
    main()