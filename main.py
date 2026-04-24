"""
README に沿ったパイプラインを実行します。

単一ステージ（既定）:
  train (config.yml) → evaluate (val) → evaluate (test)

二段階（--two-stage）:
  事前学習: Clotho-Moment (config_pretraining.yml 既定)
  → ファインチューン: CASTELLA (config.yml 既定, 事前学習の best を --resume)
  → evaluate (val) → evaluate (test)

成功時のみ slack_notice で最終メトリクス (brief) を送信します。

README との対応（事前学習込み）:
  ``python src/train.py --config config_pretraining.yml``
  → ``train_main(opt, None)``
  ``python src/train.py --config config.yml --resume results_pretraining/best_checkpoint.pth``
  → ``train_main(opt, str(pre_ckpt))`` （``pre_ckpt`` は事前学習の ``opt.ckpt_filepath``）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import BaseOptions  # noqa: E402
from evaluate import start_inference as run_eval_inference  # noqa: E402
from train import main as train_main  # noqa: E402
from create_submission import start_inference as run_private_submission  # noqa: E402


def _notify_metrics_only(
    val_brief: dict[str, Any] | None,
    test_brief: dict[str, Any] | None,
    *,
    no_test: bool,
) -> None:
    if not os.environ.get("SLACK_WEBHOOK_URL"):
        return
    try:
        from slack_notice import notify
    except ImportError:
        return
    parts: list[str] = []
    if val_brief:
        parts.append("val: " + ", ".join(f"{k}={v}" for k, v in val_brief.items()))
    if not no_test and test_brief:
        parts.append("test: " + ", ".join(f"{k}={v}" for k, v in test_brief.items()))
    if not parts:
        return
    try:
        notify("\n".join(parts))
    except Exception:
        pass


def _load_metrics_brief(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.is_file():
        return None
    try:
        with metrics_path.open(encoding="utf-8") as f:
            data = json.load(f)
        brief = data.get("brief")
        if isinstance(brief, dict):
            return dict(brief)
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _config_path(repo_root: Path, config_arg: str) -> str:
    p = Path(config_arg)
    if p.is_absolute():
        return str(p)
    return str((repo_root / p).resolve())


def _checkpoint_file(repo_root: Path, opt: Any) -> Path:
    p = Path(opt.ckpt_filepath)
    if p.is_absolute():
        return p
    return repo_root / p


def main() -> int:
    repo_root = _ROOT
    parser = argparse.ArgumentParser(
        description="train → evaluate (val) → evaluate (test)。--two-stage で Clotho 事前学習 + CASTELLA FT。",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.yml",
        help="単一ステージ時の学習・評価 YAML。--two-stage 時は CASTELLA ファインチューン用。",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        default="results/best_checkpoint.pth",
        help="評価で読むチェックポイント（既定は CASTELLA の best）。",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Clotho-Moment 事前学習のあと CASTELLA でファインチューンしてから評価する。",
    )
    parser.add_argument(
        "--pretrain-config",
        default="config_pretraining.yml",
        help="--two-stage 時の事前学習用 YAML（Clotho-Moment）。",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument(
        "--create-submission",
        action="store_true",
        help="private 用: create_submission の推論を実行",
    )
    args = parser.parse_args()

    os.chdir(repo_root)

    cfg = _config_path(repo_root, args.config)
    results_dir = repo_root / "results"
    submission_metrics = results_dir / "submission_metrics.json"

    val_brief: dict[str, Any] | None = None
    test_brief: dict[str, Any] | None = None

    try:
        if not args.skip_train:
            if args.two_stage:
                cfg_pre = _config_path(repo_root, args.pretrain_config)
                print("\n=== pretrain (Clotho-Moment) ===\n", flush=True)
                om_pre = BaseOptions(cfg_pre)
                om_pre.parse()
                train_main(om_pre.option, None)
                pre_ckpt = _checkpoint_file(repo_root, om_pre.option)

                print("\n=== finetune (CASTELLA) ===\n", flush=True)
                om_ft = BaseOptions(cfg)
                om_ft.parse()
                train_main(om_ft.option, str(pre_ckpt))
            else:
                print("\n=== train (CASTELLA only) ===\n", flush=True)
                om = BaseOptions(cfg)
                om.parse()
                train_main(om.option, None)

        if not args.skip_eval:
            print("\n=== evaluate (val) ===\n", flush=True)
            om = BaseOptions(cfg)
            om.parse()
            opt = om.option
            opt.model_path = args.model_path
            opt.eval_split_name = "val"
            run_eval_inference(opt)

            val_brief = _load_metrics_brief(submission_metrics)
            if val_brief is not None:
                try:
                    shutil.copy2(submission_metrics, results_dir / "submission_val_metrics.json")
                except OSError:
                    pass

            if not args.no_test:
                print("\n=== evaluate (test) ===\n", flush=True)
                om = BaseOptions(cfg)
                om.parse()
                opt = om.option
                opt.model_path = args.model_path
                opt.eval_split_name = "test"
                run_eval_inference(opt)
                test_brief = _load_metrics_brief(submission_metrics)

        if args.create_submission:
            print("\n=== create_submission ===\n", flush=True)
            om = BaseOptions(cfg)
            om.parse()
            opt = om.option
            opt.model_path = args.model_path
            opt.eval_split_name = "private"
            run_private_submission(opt)

        _notify_metrics_only(val_brief, test_brief, no_test=args.no_test)
        return 0

    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())