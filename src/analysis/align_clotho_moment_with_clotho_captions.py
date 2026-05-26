#!/usr/bin/env python3

"""Align Clotho-Moment queries with Clotho caption CSV rows.

The Clotho-Moment query text is derived from one of the five captions in the
Clotho caption CSV. This script normalizes both sides, finds the matching
caption, and attaches the matched CSV metadata to each JSONL record.
"""

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_caption_index(column_name):
    match = re.fullmatch(r"caption_(\d+)", column_name)
    if match is None:
        raise ValueError(f"Unexpected caption column name: {column_name}")
    return int(match.group(1))


def load_caption_index(csv_path):
    caption_index = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV file: {csv_path}")

        caption_columns = [name for name in reader.fieldnames if name.startswith("caption_")]
        if not caption_columns:
            raise ValueError(f"No caption_* columns found in {csv_path}")

        for row_idx, row in enumerate(reader):
            file_name = row.get("file_name", "")
            for caption_column in caption_columns:
                caption = (row.get(caption_column) or "").strip()
                if not caption:
                    continue
                caption_index[normalize_text(caption)].append(
                    {
                        "file_name": file_name,
                        "caption_column": caption_column,
                        "caption_index": parse_caption_index(caption_column),
                        "caption": caption,
                        "csv_row_index": row_idx,
                    }
                )

    return caption_index


def build_output_path(input_path, output_path):
    if output_path is not None:
        return Path(output_path)
    input_path = Path(input_path)
    return Path("add_data") / f"{input_path.stem}_with_clotho_match.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Clotho-Moment JSONL path")
    parser.add_argument("--csv", required=True, help="Clotho captions CSV path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to <input>_with_clotho_match.jsonl",
    )
    parser.add_argument(
        "--matched-only",
        action="store_true",
        help="Write only records that matched a caption.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any query does not match a caption.",
    )
    args = parser.parse_args()

    caption_index = load_caption_index(args.csv)
    output_path = build_output_path(args.jsonl, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    matched = 0
    unmatched_qids = []
    written = 0

    with open(args.jsonl, "r", encoding="utf-8") as in_fh, open(output_path, "w", encoding="utf-8") as out_fh:
        for line in in_fh:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total += 1

            query = record.get("query", "")
            candidates = caption_index.get(normalize_text(query), [])

            if not candidates:
                unmatched_qids.append(record.get("qid"))
                if args.strict:
                    raise ValueError(f"No caption match found for qid={record.get('qid')} query={query!r}")
                if args.matched_only:
                    continue
                record["clotho_caption_match"] = None
            else:
                matched += 1
                if len(candidates) == 1:
                    match = candidates[0]
                else:
                    match = sorted(
                        candidates,
                        key=lambda item: (item["csv_row_index"], item["caption_index"], item["caption_column"]),
                    )[0]
                    match = {
                        **match,
                        "alternatives": candidates,
                    }

                record["clotho_caption_match"] = match

            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Matched {matched}/{total} queries")
    print(f"Wrote {written} records to {output_path}")
    if unmatched_qids:
        preview = ", ".join(map(str, unmatched_qids[:10]))
        suffix = "" if len(unmatched_qids) <= 10 else f" ... (+{len(unmatched_qids) - 10} more)"
        print(f"Unmatched qids: {preview}{suffix}")


if __name__ == "__main__":
    main()