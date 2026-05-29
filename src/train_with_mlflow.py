import os
import time
import json
import pprint
import random
import argparse
import copy
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from easydict import EasyDict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import BaseOptions
from dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from evaluate import eval_epoch, start_inference, setup_model

from basic_utils import AverageMeter, dict_to_markdown, write_log, save_checkpoint, rename_latest_to_best
from model_utils import count_parameters, ModelEMA

import logging
import mlflow
import git

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def get_git_info():
    """Get git commit hash and branch name"""
    try:
        repo = git.Repo('.')
        return {
            "git_commit": repo.head.commit.hexsha,
            "git_branch": repo.active_branch.name,
        }
    except Exception as e:
        logger.warning(f"Could not get git info: {e}")
        return {"git_commit": "unknown", "git_branch": "unknown"}


def train_epoch(
        model,
        criterion,
        train_loader,
        optimizer,
        opt,
        epoch_i
    ):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    # init meters
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)

        outputs = model(**model_inputs, targets=targets) if opt.model_name == 'cg_detr' else model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)

        optimizer.zero_grad()
        losses.backward()

        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * criterion.weight_dict[k] if k in criterion.weight_dict else float(v))

    write_log(opt, epoch_i, loss_meters)
    
    # MLflow: Log training metrics
    for metric_name, meter in loss_meters.items():
        mlflow.log_metric(f"train_{metric_name}", meter.avg, step=epoch_i)


def train(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        train_dataset,
        val_dataset,
        opt
    ):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    save_submission_filename = "latest_{}_val_preds.jsonl".format(opt.dset_name)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )

    if opt.model_ema:
        logger.info("Using model EMA...")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    prev_best_score = 0
    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i)
        lr_scheduler.step()
        
        # MLflow: Log learning rate
        mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch_i)

        if opt.model_ema:
            model_ema.update(model)

        if (epoch_i + 1) % opt.eval_epoch_interval == 0:
            with torch.no_grad():
                if opt.model_ema:
                    metrics, eval_loss_meters, latest_file_paths = \
                        eval_epoch(model_ema.module, val_dataset, opt, save_submission_filename, criterion)
                else:
                    metrics, eval_loss_meters, latest_file_paths = \
                        eval_epoch(model, val_dataset, opt, save_submission_filename, criterion)

            write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode='val')
            logger.info("metrics {}".format(pprint.pformat(metrics["brief"], indent=4)))
            
            # MLflow: Log validation loss metrics
            for metric_name, meter in eval_loss_meters.items():
                mlflow.log_metric(f"val_{metric_name}", meter.avg, step=epoch_i)
            
            # MLflow: Log validation performance metrics
            if metrics:
                for metric_name, metric_value in metrics["brief"].items():
                    # Replace @ with _at_ for MLflow compatibility
                    sanitized_name = metric_name.replace("@", "_at_")
                    mlflow.log_metric(f"val_{sanitized_name}", metric_value, step=epoch_i)

            stop_score = metrics["brief"]["MR-full-R1@0.7"]

            if stop_score > prev_best_score:
                prev_best_score = stop_score
                save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
                logger.info("The checkpoint file has been updated.")
                
                # MLflow: Log best checkpoint and metrics (sanitize metric names)
                mlflow.log_metric("best_epoch", epoch_i, step=epoch_i)
                mlflow.log_metric("best_score_R1_at_0.7", stop_score, step=epoch_i)
                
                # Log checkpoint as artifact
                for file_path in latest_file_paths:
                    if file_path.endswith(".pth"):
                        mlflow.log_artifact(file_path, artifact_path="checkpoints")
                    elif file_path.endswith("_metrics.json"):
                        mlflow.log_artifact(file_path, artifact_path="val_metrics")
                    elif file_path.endswith(".jsonl"):
                        mlflow.log_artifact(file_path, artifact_path="val_predictions")
                
                rename_latest_to_best(latest_file_paths)


def main(opt, resume=None, exp_name=None):
    logger.info("Setup config, data and model...")
    set_seed(opt.seed)
    
    # MLflow setup
    mlflow.set_experiment("DCASE-Task6-Training")
    
    with mlflow.start_run(run_name=exp_name):
        # Log git info
        git_info = get_git_info()
        mlflow.log_param("git_commit", git_info["git_commit"])
        mlflow.log_param("git_branch", git_info["git_branch"])
        
        # Log model architecture and training config
        mlflow.log_param("model_name", opt.model_name)
        mlflow.log_param("dataset_name", opt.dset_name)
        mlflow.log_param("batch_size", opt.bsz)
        mlflow.log_param("num_epochs", opt.n_epoch)
        mlflow.log_param("learning_rate", opt.lr)
        mlflow.log_param("weight_decay", opt.wd)
        mlflow.log_param("lr_drop", opt.lr_drop)
        mlflow.log_param("grad_clip", opt.grad_clip)
        mlflow.log_param("max_q_l", opt.max_q_l)
        mlflow.log_param("max_a_l", opt.max_a_l)
        mlflow.log_param("clip_length", opt.clip_length)
        mlflow.log_param("model_ema", opt.model_ema)
        if opt.model_ema:
            mlflow.log_param("ema_decay", opt.ema_decay)

        # dataset & data loader
        train_q_feat_dir = getattr(opt, "t_para_feat_dir", None) or opt.t_feat_dir
        train_q_feat_sampling = "random" if getattr(opt, "t_para_feat_dir", None) else "single"
        dataset_config = EasyDict(
            data_path=opt.train_path,
            ctx_mode=opt.ctx_mode,
            a_feat_dir=opt.a_feat_dir,
            q_feat_dir=train_q_feat_dir,
            q_feat_type="last_hidden_state",
            a_feat_type=opt.a_feat_type,
            q_feat_sampling=train_q_feat_sampling,
            max_q_l=opt.max_q_l,
            max_a_l=opt.max_a_l,
            clip_len=opt.clip_length,
            max_windows=opt.max_windows,
            span_loss_type=opt.span_loss_type,
            load_labels=True,
        )

        train_dataset = StartEndDataset(**dataset_config)
        copied_eval_config = copy.deepcopy(dataset_config)
        copied_eval_config.data_path = opt.val_path
        copied_eval_config.q_feat_dir = opt.t_feat_dir
        copied_eval_config.q_feat_sampling = "single"
        eval_dataset = StartEndDataset(**copied_eval_config)

        # prepare model
        model, criterion, optimizer, lr_scheduler = setup_model(opt)

        logger.info(f"Model {model}")
        count_parameters(model, verbose=True)

        if resume is not None:
            checkpoint = torch.load(resume, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            logger.info("Loaded model checkpoint: {}".format(resume))

        logger.info("Start Training...")

        # start training
        train(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            train_dataset,
            eval_dataset,
            opt
        )
        
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='config path')
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        help="specify model path for fine-tuning. If None, train the model from scratch.",
    )
    parser.add_argument(
        "--exp_name",
        "-e",
        type=str,
        default=None,
        help="experiment name for MLflow",
    )
    args = parser.parse_args()
    option_manager = BaseOptions(args.config)
    option_manager.parse()
    opt = option_manager.option
    main(opt, args.resume, args.exp_name)
