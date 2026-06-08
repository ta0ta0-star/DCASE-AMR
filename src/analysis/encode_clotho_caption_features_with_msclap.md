# `encode_clotho_caption_features_with_msclap.py` の実行方法

このメモは、`src/analysis/encode_clotho_caption_features_with_msclap.py` を使って Clotho caption のテキスト特徴量を NPZ に出力する手順をまとめたものです。

このスクリプトは、`add_data/clotho_caption_manifest_train.jsonl` のような 1 行 1 caption の manifest を読み、M2D-CLAP の BERT text encoder で token-level の `last_hidden_state` を保存します。出力される NPZ は `qid{qid}_caption{caption_index}.npz` という名前で、各ファイルには `last_hidden_state` のみが入ります。

## 前提

- Python 実行環境は `/home/y255618g/m2d/.venv` を使う
- M2D-CLAP のチェックポイントは `/data/y255618g/m2d/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth`
- M2D リポジトリは `/home/y255618g/m2d`

## 本番実行

CUDA の特定 GPU を使いたい場合は `--device cuda:1` のように指定します。

```bash
/home/y255618g/m2d/.venv/bin/python /home/y255618g/DCASE-AMR/src/analysis/encode_clotho_caption_features_with_msclap.py \
  --manifest /home/y255618g/DCASE-AMR/add_data/clotho_caption_manifest_train.jsonl \
  --output-dir /path/to/output_dir \
  --weight /data/y255618g/m2d/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth \
  --m2d-root /home/y255618g/m2d \
  --device cuda:1 \
  --progress
```

`--limit` は付けないと全件処理になります。`--progress` を付けると tqdm が使える環境では進捗バーを表示します。

## smoke test

動作確認だけしたい場合は `--limit 2` を付けます。

```bash
/home/y255618g/m2d/.venv/bin/python /home/y255618g/DCASE-AMR/src/analysis/encode_clotho_caption_features_with_msclap.py \
  --manifest /tmp/sample_manifest.jsonl \
  --output-dir /tmp/clotho_feats_test \
  --weight /data/y255618g/m2d/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth \
  --m2d-root /home/y255618g/m2d \
  --device auto \
  --limit 2
```

## 出力内容の確認

NPZ の中身は次のように確認できます。

```bash
/home/y255618g/m2d/.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np

out_dir = Path('/tmp/clotho_feats_test')
for path in sorted(out_dir.glob('*.npz')):
    with np.load(path) as data:
        print(path.name, data.files)
        arr = data['last_hidden_state']
        print(arr.shape, arr.dtype)
        print(arr.reshape(-1)[:3])
PY
```

## 実装上の注意

- 出力は caption ごとの token-level feature なので、caption 長によって `shape[0]` は変わる
- 保存キーは `last_hidden_state` 固定
- 既存の `msclap` 版と違い、現行実装は M2D の tokenizer と BERT encoder を直接使う
