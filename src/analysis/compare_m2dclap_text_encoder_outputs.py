#!/usr/bin/env python3

"""Compare M2D-CLAP text encoder outputs from two loading paths.

This script loads the same checkpoint through RuntimeM2D and PortableM2D,
feeds one caption through both text encoders, and reports the maximum
absolute error between the resulting `last_hidden_state` arrays.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from encode_clotho_caption_features_with_m2dclap import encode_caption, load_m2d_text_encoder


def load_portable_m2d_text_encoder(weight_path, m2d_root, device):
    sys.path.insert(0, str(Path(m2d_root).resolve()))
    from examples.portable_m2d import PortableM2D

    model = PortableM2D(str(Path(weight_path).resolve()))
    model = model.to(device)
    model.eval()
    model.get_clap_text_encoder()

    text_encoder = model.text_encoder
    tokenizer = text_encoder.tokenizer
    bert_model = text_encoder.text_encoder
    return tokenizer, bert_model


def compare_caption(caption, runtime_tokenizer, runtime_model, portable_tokenizer, portable_model, device):
    runtime_feat = encode_caption(runtime_tokenizer, runtime_model, caption, device)
    portable_feat = encode_caption(portable_tokenizer, portable_model, caption, device)

    if runtime_feat.shape != portable_feat.shape:
        raise RuntimeError(
            f"shape mismatch: runtime={runtime_feat.shape} portable={portable_feat.shape}"
        )

    diff = np.abs(runtime_feat - portable_feat)
    max_abs_error = float(diff.max()) if diff.size else 0.0
    mean_abs_error = float(diff.mean()) if diff.size else 0.0
    return runtime_feat, portable_feat, max_abs_error, mean_abs_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption", required=True, help="Caption text to encode once")
    parser.add_argument("--weight", required=True, type=Path, help="Path to the M2D-CLAP checkpoint")
    parser.add_argument("--m2d-root", type=Path, default=Path("/home/y255618g/m2d"), help="Path to the local m2d repository root")
    parser.add_argument("--device", default="auto", help="Device to run on: auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    runtime_tokenizer, runtime_model = load_m2d_text_encoder(args.weight, args.m2d_root, device)
    portable_tokenizer, portable_model = load_portable_m2d_text_encoder(args.weight, args.m2d_root, device)

    runtime_feat, portable_feat, max_abs_error, mean_abs_error = compare_caption(
        args.caption,
        runtime_tokenizer,
        runtime_model,
        portable_tokenizer,
        portable_model,
        device,
    )

    print(f"device: {device}")
    print(f"caption: {args.caption}")
    print(f"runtime shape: {runtime_feat.shape}, dtype: {runtime_feat.dtype}")
    print(f"portable shape: {portable_feat.shape}, dtype: {portable_feat.dtype}")
    print(f"max_abs_error: {max_abs_error:.10e}")
    print(f"mean_abs_error: {mean_abs_error:.10e}")


if __name__ == "__main__":
    main()