#!/usr/bin/env python3

"""Export one caption record per paraphrase for Clotho-Moment pretraining.

This reads an aligned Clotho-Moment JSONL and the original Clotho caption CSV,
then emits a JSONL file with five caption texts per source example.

The output is suitable as input to an external text embedding pipeline.
"""

import argparse
import csv
import json
from pathlib import Path


def load_csv_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV file: {csv_path}")
        return list(reader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Aligned Clotho-Moment JSONL path")
    parser.add_argument("--csv", required=True, help="Clotho captions CSV path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    csv_rows = load_csv_rows(args.csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0

    with open(args.jsonl, "r", encoding="utf-8") as in_fh, open(output_path, "w", encoding="utf-8") as out_fh:
        for line in in_fh:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total += 1

            match = record.get("clotho_caption_match")
            if not match:
                continue

            row_index = match.get("csv_row_index")
            if row_index is None:
                continue
            if row_index < 0 or row_index >= len(csv_rows):
                raise IndexError(f"csv_row_index out of range for qid={record.get('qid')}: {row_index}")

            row = csv_rows[row_index]
            for caption_index in range(1, 6):
                caption_key = f"caption_{caption_index}"
                caption = (row.get(caption_key) or "").strip()
                if not caption:
                    continue

                output_record = {
                    "qid": record.get("qid"),
                    "source_qid": record.get("qid"),
                    "file_name": row.get("file_name", ""),
                    "caption_index": caption_index,
                    "caption_key": caption_key,
                    "caption": caption,
                    "query": caption,
                    "duration": record.get("duration"),
                    "vid": record.get("vid"),
                    "relevant_windows": record.get("relevant_windows"),
                    "fg_dB": record.get("fg_dB"),
                }
                out_fh.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                written += 1

    print(f"Processed {total} aligned records")
    print(f"Wrote {written} caption records to {output_path}")


if __name__ == "__main__":
    main()
