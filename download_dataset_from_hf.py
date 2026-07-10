#!/usr/bin/env python3

"""Download a single file from a Hugging Face dataset repository.

This script is meant for dataset artifacts like tar archives stored in a
dataset repo. It uses `huggingface_hub` so the download works against the
canonical resolved file URL instead of scraping the web UI.
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--repo-id",
		default="ta0ta00h/m2dclap-features",
		help="Hugging Face repository ID",
	)
	parser.add_argument(
		"--filename",
		default="clotho_clap_text.tar",
		help="File name inside the dataset repository (kept for backward compatibility)",
	)

	parser.add_argument(
		"--filenames",
		nargs="+",
		help="One or more file names to download from the repository",
	)
	parser.add_argument(
		"--manifest",
		type=Path,
		help="Path to a local file listing filenames to download (one per line)",
	)
	parser.add_argument(
		"--jobs",
		type=int,
		default=1,
		help="Number of parallel download jobs (default: 1)",
	)
	parser.add_argument(
		"--repo-type",
		default="dataset",
		choices=["dataset", "model", "space"],
		help="Hugging Face repo type",
	)
	parser.add_argument(
		"--local-dir",
		type=Path,
		default=Path("/data/y255618g/dcase2026_task6/"),
		help="Directory to store the downloaded file",
	)
	parser.add_argument(
		"--local-dir-use-symlinks",
		action="store_true",
		help="Use symlinks in the local directory when possible",
	)
	args = parser.parse_args()

	args.local_dir.mkdir(parents=True, exist_ok=True)

	# Build list of filenames to download
	if args.filenames:
		filenames = args.filenames
	elif args.manifest:
		with args.manifest.open("r", encoding="utf-8") as f:
			filenames = [l.strip() for l in f if l.strip()]
	else:
		filenames = [args.filename]

	def download_one(filename: str) -> str:
		try:
			return hf_hub_download(
				repo_id=args.repo_id,
				repo_type=args.repo_type,
				filename=filename,
				local_dir=str(args.local_dir),
				local_dir_use_symlinks=args.local_dir_use_symlinks,
			)
		except Exception as e:
			# Return an error message prefixed so caller can distinguish
			return f"ERROR:{filename}:{e}"

	results = []
	if args.jobs and args.jobs > 1 and len(filenames) > 1:
		with ThreadPoolExecutor(max_workers=args.jobs) as ex:
			fut_to_name = {ex.submit(download_one, fn): fn for fn in filenames}
			for fut in as_completed(fut_to_name):
				res = fut.result()
				results.append(res)
	else:
		for fn in filenames:
			results.append(download_one(fn))

	# Print results: successful local paths or errors
	for r in results:
		if isinstance(r, str) and r.startswith("ERROR:"):
			# Format: ERROR:filename:exception
			parts = r.split(":", 2)
			msg = parts[2] if len(parts) > 2 else "unknown error"
			print(f"Failed: {parts[1]} -> {msg}", file=sys.stderr)
		else:
			print(r)


if __name__ == "__main__":
	main()
