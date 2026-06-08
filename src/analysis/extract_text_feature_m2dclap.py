"""M2D-CLAP版 テキスト特徴量抽出スクリプト
MS-CLAPと同じフォーマット（key='last_hidden_state', 768次元）で出力する。
M2D-CLAP 2025はBERT-baseをテキストエンコーダに使う。

使い方:
    cd ~/PDS1/m2d
    python ~/PDS1/DCASE-AMR/extract_text_feature_m2dclap.py \
        ~/PDS1/castella/json/en/train.json \
        --weight m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth \
        --output_dir ~/PDS1/lighthouse/features/castella/clap_text

    # val, testも同様に実行
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path, help="CASTELLAアノテーションJSON")
    parser.add_argument("--weight", type=str, required=True, help="M2D-CLAPの重みファイルパス")
    parser.add_argument("--output_dir", type=Path, default=Path("features/castella/clap_text"))
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    if not args.json_path.exists():
        print(f"{args.json_path} does not exist.")
        return

    # M2D-CLAPモデルのロード（テキストエンコーダ取得用）
    m2d_root = str(Path(args.weight).resolve().parent.parent)
    sys.path.insert(0, m2d_root)
    from examples.portable_m2d import PortableM2D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PortableM2D(args.weight)
    model.eval()
    model.get_clap_text_encoder()

    # テキストエンコーダのコンポーネントを取得
    text_enc = model.text_encoder.to(device)
    tokenizer = text_enc.tokenizer
    bert_model = text_enc.text_encoder
    print(f"Inference on {device}")

    # アノテーション読み込み
    with open(args.json_path) as f:
        data = json.load(f)

    for item in tqdm(data):
        yid = item["yid"]
        for idx, moment in enumerate(item["moments"]):
            caption = moment["local_caption"]
            path_feats = args.output_dir / f"qid{yid}_{idx + 1}.npz"

            if path_feats.exists():
                continue

            with torch.no_grad():
                inputs = tokenizer(
                    caption,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = bert_model(**inputs)
                hidden_state = outputs.last_hidden_state  # (1, num_tokens, 768)

                # attention_maskでpadding除去（padding=Falseなので基本不要だが安全のため）
                mask = inputs["attention_mask"]
                seq_len = mask.sum().item()
                feat = hidden_state[0, :seq_len].cpu().numpy()  # (num_tokens, 768)

            np.savez(path_feats, last_hidden_state=feat)

    print("Done.")


if __name__ == "__main__":
    main()
