#!/usr/bin/env python3

"""Attach an existing results directory to an MLflow run.

This is useful when you already have a `results_pretraining/` directory and want
to preserve the config, git info, and result artifacts under a reproducible
MLflow run name.
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
import yaml
import git


def read_text_if_exists(path: Path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def get_git_info(repo_path: Path):
    try:
        repo = git.Repo(repo_path)
        branch = None
        try:
            branch = repo.active_branch.name
        except Exception:
            branch = "detached"
        return {
            "git_commit": repo.head.commit.hexsha,
            "git_branch": branch,
        }
    except Exception:
        return {
            "git_commit": "unknown",
            "git_branch": "unknown",
        }


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Existing results directory to attach")
    parser.add_argument("--config", required=True, help="Training config used for the run")
    parser.add_argument("--experiment-name", default="DCASE-Task6-Training", help="MLflow experiment name")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--tracking-uri", default=None, help="Optional MLflow tracking URI")
    parser.add_argument("--repo-root", default=".", help="Repository root for git metadata")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    config_path = Path(args.config)
    repo_root = Path(args.repo_root)

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    config = load_config(config_path)
    git_info = get_git_info(repo_root)

    payload = {
        "results_dir": str(results_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "git_info": git_info,
        "config": config,
    }

    summary_path = results_dir / "mlflow_bundle.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("results_dir", str(results_dir.resolve()))
        mlflow.log_param("config_path", str(config_path.resolve()))
        mlflow.log_param("git_commit", git_info["git_commit"])
        mlflow.log_param("git_branch", git_info["git_branch"])

        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                mlflow.log_param(f"config.{key}", value)

        mlflow.log_artifact(str(config_path), artifact_path="config")

        for artifact_name in [
            "best_checkpoint.pth",
            "best_clotho-moment_val_preds.jsonl",
            "best_clotho-moment_val_preds_metrics.json",
            "latest_clotho-moment_val_preds.jsonl",
            "latest_clotho-moment_val_preds_metrics.json",
            "train.log",
            "val.log",
            "mlflow_bundle.json",
        ]:
            artifact_path = results_dir / artifact_name
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path="results_pretraining")

        mlflow.set_tag("source", "existing_results")
        mlflow.set_tag("results_dir", str(results_dir.resolve()))

        print(f"Logged existing results from {results_dir} to MLflow run {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()