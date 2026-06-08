"""M2D-CLAP版 音声特徴量抽出スクリプト
MS-CLAPと同じフォーマット（key='features', 1fps, 768次元）で出力する。

使い方:
    cd ~/PDS1/m2d
    python ~/PDS1/DCASE-AMR/extract_audio_feature_m2dclap.py \
        ~/PDS1/CASTELLA-audio/download/audio \
        --weight m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth \
        --output_dir ~/PDS1/lighthouse/features/castella/clap
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", type=Path, help="wavファイルのディレクトリ")
    parser.add_argument("--weight", type=str, required=True, help="M2D-CLAPの重みファイルパス")
    parser.add_argument("--output_dir", type=Path, default=Path("features/castella/clap"))
    parser.add_argument("--max_duration", type=float, default=300.0, help="最大秒数")
    parser.add_argument("--target_fps", type=float, default=1.0, help="出力フレームレート")
    parser.add_argument("--sr", type=int, default=16000, help="サンプリングレート")
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    # M2D-CLAPモデルのロード
    # m2dリポジトリのルートからimportする必要がある
    m2d_root = str(Path(args.weight).resolve().parent.parent)
    sys.path.insert(0, m2d_root)
    from examples.portable_m2d import PortableM2D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PortableM2D(args.weight)
    model.cfg.flat_features = True
    model.eval()
    model = model.to(device)
    print(f"Inference on {device}")

    # まずフレームレートを計算（10秒のダミーで測定）
    with torch.no_grad():
        dummy = torch.randn(1, 10 * args.sr).to(device)
        dummy_out = model.encode(dummy)
        native_fps = dummy_out.shape[1] / 10.0
        print(f"M2D-CLAP native fps: {native_fps:.1f}, target fps: {args.target_fps}")

    list_wav = sorted(args.audio_dir.glob("*.wav"))
    print(f"Found {len(list_wav)} wav files")

    skipped = 0
    for path_wav in tqdm(list_wav):
        path_feats = args.output_dir / f"{path_wav.stem}.npz"
        if path_feats.exists():
            continue

        try:
            audio, sr = librosa.load(str(path_wav), sr=args.sr, mono=True)
        except Exception as e:
            print(f"Error loading {path_wav.name}: {e}")
            skipped += 1
            continue

        # 最大秒数でトランケート
        max_samples = int(args.max_duration * args.sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        duration = len(audio) / args.sr
        audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)

        with torch.no_grad():
            # encode() → (1, T_native, 768) at native_fps
            features = model.encode(audio_tensor)  # (1, T, 768)
            features = features.squeeze(0).cpu().numpy()  # (T, 768)

        # native_fpsからtarget_fps(1fps)にダウンサンプル
        n_native = features.shape[0]
        n_target = int(duration * args.target_fps)
        if n_target < 1:
            n_target = 1

        # 均等にフレームを分割して平均プーリング
        downsampled = np.zeros((n_target, features.shape[1]), dtype=np.float32)
        for i in range(n_target):
            start = int(i * n_native / n_target)
            end = int((i + 1) * n_native / n_target)
            if end > n_native:
                end = n_native
            if start >= end:
                start = max(0, end - 1)
            downsampled[i] = features[start:end].mean(axis=0)

        np.savez(path_feats, features=downsampled)

    print(f"Done. Skipped: {skipped}")


if __name__ == "__main__":
    main()
