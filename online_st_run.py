import os
from typing import Literal
from src.datasets.ukbiobank_dataset import UkBiobankDatasetArgs
from src.args.yaml_config import YamlConfig
from src.models.auto_sam_model import AutoSamModel, AutoSamModelArgs
from tqdm import tqdm
import numpy as np
import torch
from src.args.yaml_config import YamlConfig
from pathlib import Path
import argparse
import subprocess
from datetime import datetime
import shlex

yaml_config = YamlConfig().config
# Mask output path from argparse
dir_suffix = f"{datetime.now():%Y-%m-%d_%H#%M#%S}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse arguments for out_dir_name and teacher_checkpoint."
    )
    parser.add_argument("--st_epochs", type=int, help="Number of epochs")
    parser.add_argument(
        "--n_teacher_samples",
        type=int,
        help="Number of samples to use for teacher",
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        help="Path to teacher checkpoint, if not given will train teacher",
    )
    parser.add_argument(
        "--st_learning_rate", type=float, help="Learning rate for self training"
    )
    parser.add_argument(
        "--ft_learning_rate", type=float, help="Learning rate for finetuning"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


def exec(cmd: str):
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)

    if result.returncode == 0:
        print("Command Output:\n", result.stdout)
    else:
        raise Exception(f"Error while executing '{cmd}':\n", result.stderr)


def run_teacher_prompt_encoder_pretraining(n_teacher_samples: str):
    teacher_experiment_subdir_name = (
        f"online_st_teacher_{n_teacher_samples}_samples_{dir_suffix}"
    )
    teacher_checkpoint_dir = (
        Path(yaml_config.results_dir)
        / "multi_ds_vessel_experiment"
        / teacher_experiment_subdir_name
    )

    print(f"Training teacher prompt encoder...")
    tags = '["PromptEncoder", "Finetuning", "TeacherTraining", "VesselSeg"]'
    # TODO: --wandb_tags={shlex.quote(tags)} --use_wandb=true --wandb_experiment_name=gt_vessels_onlineST_teacher_{n_teacher_samples}_samples
    train_teacher_cmd = f"python run.py --experiment_id=multi_ds_vessel --sam_model=vit_b --learning_rate=0.0003 --prompt_encoder_lr=0.003 --image_encoder_lr=0.0 --mask_decoder_lr=0.0 --batch_size=3 --epochs=1 --weight_decay=1e-4 --early_stopping_patience=10 --visualize_n_segmentations=5 --gamma=0.9 --step_size=1 --best_model_metric=IoU --minimize_best_model_metric=false  --amp=true --results_subdir_name={teacher_experiment_subdir_name}"

    if n_teacher_samples != "all":
        train_teacher_cmd += f" --limit_train_samples={n_teacher_samples}"
    exec(train_teacher_cmd)

    teacher_dirs = os.listdir(teacher_checkpoint_dir)
    assert (
        len(teacher_dirs) == 1
    ), f"Expected 1 dir in {teacher_checkpoint_dir}, got {len(teacher_dirs)}"
    return teacher_checkpoint_dir / teacher_dirs[0] / "model.pt"


def run_teacher_pretraining(n_teacher_samples: str):
    teacher_experiment_subdir_name = (
        f"online_st_teacher_{n_teacher_samples}_samples_{dir_suffix}"
    )
    teacher_checkpoint_dir = (
        Path(yaml_config.results_dir)
        / "multi_ds_vessel_experiment"
        / teacher_experiment_subdir_name
    )

    print(f"Training teacher...")
    tags = '["FullModel", "Finetuning", "TeacherTraining", "VesselSeg", "GT"]'
    # TODO: --wandb_tags={shlex.quote(tags)} --use_wandb=true --wandb_experiment_name=gt_vessels_onlineST_teacher_{n_teacher_samples}_samples
    train_teacher_cmd = f"""python run.py --experiment_id=multi_ds_vessel --sam_model=vit_b 
    --learning_rate=0.0003 --prompt_encoder_lr=0.0001 --image_encoder_lr=0.00001
    --mask_decoder_lr=0.0001 --batch_size=3 --epochs=1 --weight_decay=1e-4 --early_stopping_patience=10
    --visualize_n_segmentations=5 --gamma=0.95 --step_size=3 --best_model_metric=IoU
    --minimize_best_model_metric=false --prompt_encoder_checkpoint={teacher_checkpoint_dir}/prompt_encoder.pt
    --amp=true --results_subdir_name={teacher_experiment_subdir_name}"""

    if n_teacher_samples != "all":
        train_teacher_cmd += f" --limit_train_samples={n_teacher_samples}"
    exec(train_teacher_cmd)

    teacher_dirs = os.listdir(teacher_checkpoint_dir)
    assert (
        len(teacher_dirs) == 1
    ), f"Expected 1 dir in {teacher_checkpoint_dir}, got {len(teacher_dirs)}"
    return teacher_checkpoint_dir / teacher_dirs[0] / "model.pt"


def run_student_st(teacher_checkpoint: str, epochs: int, learning_rate: float):
    student_experiment_subdir_name = (
        f"onlineST_student_{n_teacher_samples}_samples_{dir_suffix}"
    )
    print("Teacher checkpoint:", teacher_checkpoint)
    # tags = '["FullModel", "OnlineST", "VesselSeg", "PseudoLabels"]'
    exec(
        f"python run.py --experiment_id=self_learning_experiment --limit=100 --sam_model=vit_b --learning_rate={str(learning_rate)} --batch_size=20 --epochs={str(epochs)} --weight_decay=1e-4 --gamma=0.9 --step_size=2 --best_model_metric=IoU --minimize_best_model_metric=false --drive_test_equals_val=false --amp=true --augment_train=true --ema_decay=0.95 --from_checkpoint={teacher_checkpoint} --results_subdir_name={student_experiment_subdir_name} --return_best_model=false"
    )
    student_checkpoint_dir = (
        "/dhc/home/leon.hermann/MP/mp-lippert-experiments"
        / Path(yaml_config.results_dir)
        / "self_learning_experiment"
        / student_experiment_subdir_name
    )
    student_dirs = os.listdir(student_checkpoint_dir)
    assert (
        len(student_dirs) == 1
    ), f"Expected 1 dir in {student_checkpoint_dir}, got {len(student_dirs)}"
    return student_checkpoint_dir / student_dirs[0] / "model.pt"


def run_full_fine_tuning(
    prefix: Literal["noST", "onlineST"],
    n_teacher_samples: str,
    checkpoint_path: str,
    learning_rate: float,
):
    subdir_name = f"{prefix}_fft_{n_teacher_samples}_samples_{dir_suffix}"
    tags = '["FullModel", "FinalFinetuning", "GT", "VesselSeg"]'
    cmd = f"python run.py --experiment_id=multi_ds_vessel --wandb_tags={shlex.quote(tags)} --wandb_experiment_name=vessels_gt_{prefix}_fft_{n_teacher_samples}_samples --use_wandb=true --sam_model=vit_b --learning_rate={str(learning_rate)} --batch_size=4 --epochs=1 --weight_decay=1e-4 --gamma=0.9 --step_size=2 --best_model_metric=IoU --minimize_best_model_metric=false --from_checkpoint={checkpoint_path} --drive_test_equals_val=false --amp=true --results_subdir_name={subdir_name}"
    if n_teacher_samples != "all":
        cmd += f" --limit_train_samples={n_teacher_samples}"
    exec(cmd)
    checkpoint_dir = (
        "/dhc/home/leon.hermann/MP/mp-lippert-experiments"
        / Path(yaml_config.results_dir)
        / "multi_ds_vessel_experiment"
        / subdir_name
    )
    dirs = os.listdir(checkpoint_dir)
    assert len(dirs) == 1, f"Expected 1 dir in {checkpoint_dir}, got {len(dirs)}"
    return checkpoint_dir / dirs[0] / "model.pt"


if __name__ == "__main__":
    args = parse_args()

    n_teacher_samples = str(args.n_teacher_samples) if args.n_teacher_samples else "all"
    if args.teacher_checkpoint:
        teacher_checkpoint = args.teacher_checkpoint
    else:
        teacher_prompt_encoder_checkpoint = run_teacher_prompt_encoder_pretraining(
            n_teacher_samples
        )
        teacher_checkpoint = run_teacher_pretraining(n_teacher_samples)
        print(f"Teacher trained and saved in {teacher_checkpoint}.")

    print("Training student...")
    student_checkpoint = run_student_st(
        str(teacher_checkpoint), args.st_epochs, args.st_learning_rate
    )

    print("Training FFT onlineST")
    run_full_fine_tuning(
        "onlineST", n_teacher_samples, str(student_checkpoint), args.ft_learning_rate
    )
