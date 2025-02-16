import os
from typing import Literal, Optional
from src.args.yaml_config import YamlConfig
from src.args.yaml_config import YamlConfig
from pathlib import Path
import argparse
import subprocess
from datetime import datetime
import shlex
import json

yaml_config = YamlConfig().config
# Mask output path from argparse
dir_suffix = f"{datetime.now():%Y-%m-%d_%H#%M#%S}"
debug = False
debug_prefix = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse arguments for out_dir_name and teacher_checkpoint."
    )
    parser.add_argument(
        "--n_teacher_samples",
        type=int,
        help="Number of samples to use for teacher",
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


def run_teacher_pretraining1(n_teacher_samples: str):
    teacher_experiment_subdir_name = (
        f"offlineST_teacher_step1_{n_teacher_samples}_samples_{dir_suffix}"
    )
    teacher_checkpoint_dir = (
        Path(yaml_config.results_dir)
        / "multi_ds_vessel_experiment"
        / teacher_experiment_subdir_name
    )
    tags = f'["OfflineST","TeacherTraining","TeacherTraining1","VesselSeg","GT","PromptEncoder", "{n_teacher_samples}_samples"]'

    print(f"Training teacher prompt encoder only...")
    train_teacher_cmd = f"python run.py --experiment_id=multi_ds_vessel --sam_model=vit_b --learning_rate=0.003 --prompt_encoder_lr=0.003 --batch_size=8 --epochs=30 --weight_decay=1e-4 --early_stopping_patience=10 --visualize_n_segmentations=5 --gamma=0.9 --step_size=1 --best_model_metric=IoU --minimize_best_model_metric=false --drive_test_equals_val=false --use_wandb=true --wandb_experiment_name={debug_prefix}gt_vessels_offlineST_teacher1_{n_teacher_samples}_samples --amp=true --results_subdir_name={teacher_experiment_subdir_name} --wandb_tags={shlex.quote(tags)}"
    if n_teacher_samples != "all":
        train_teacher_cmd += f" --limit_train_samples={n_teacher_samples}"
    exec(train_teacher_cmd)

    teacher_dirs = os.listdir(teacher_checkpoint_dir)
    assert (
        len(teacher_dirs) == 1
    ), f"Expected 1 dir in {teacher_checkpoint_dir}, got {len(teacher_dirs)}"
    return teacher_checkpoint_dir / teacher_dirs[0] / "model.pt"


def run_teacher_pretraining2(n_teacher_samples: str, teacher_checkpoint: str):
    teacher_experiment_subdir_name = (
        f"offlineST_teacher_step2_{n_teacher_samples}_samples_{dir_suffix}"
    )
    teacher_checkpoint_dir = (
        Path(yaml_config.results_dir)
        / "multi_ds_vessel_experiment"
        / teacher_experiment_subdir_name
    )
    tags = f'["OfflineST","TeacherTraining","TeacherTraining2","VesselSeg","GT","FullModel", "{n_teacher_samples}_samples"]'

    print(f"Training teacher full model...")
    train_teacher_cmd = f"python run.py --experiment_id=multi_ds_vessel --sam_model=vit_b --learning_rate=0.0003 --prompt_encoder_lr=0.0001 --image_encoder_lr=0.00001 --mask_decoder_lr=0.0001 --batch_size=3 --epochs=50 --weight_decay=1e-4 --early_stopping_patience=10 --visualize_n_segmentations=5 --gamma=0.95 --step_size=3 --best_model_metric=IoU --minimize_best_model_metric=false --drive_test_equals_val=false --use_wandb=true --wandb_experiment_name={debug_prefix}gt_vessels_offlineST_teacher_{n_teacher_samples}_samples --amp=true --results_subdir_name={teacher_experiment_subdir_name} --wandb_tags={shlex.quote(tags)} --from_checkpoint={teacher_checkpoint}"
    if n_teacher_samples != "all":
        train_teacher_cmd += f" --limit_train_samples={n_teacher_samples}"
    exec(train_teacher_cmd)

    teacher_dirs = os.listdir(teacher_checkpoint_dir)
    assert (
        len(teacher_dirs) == 1
    ), f"Expected 1 dir in {teacher_checkpoint_dir}, got {len(teacher_dirs)}"
    return teacher_checkpoint_dir / teacher_dirs[0] / "model.pt"


def generate_pseudo_labels(
    n_teacher_samples: str, teacher_checkpoint: str, limit: Optional[int] = None
):
    pseudo_labels_dir_name = f"teacher_{n_teacher_samples}_samples"
    cmd = f"python src/wild-west/generate_labels.py --out_dir_name={pseudo_labels_dir_name} --teacher_checkpoint={teacher_checkpoint}"
    if limit != None:
        cmd += f" --limit={limit}"
    exec(cmd)
    return (
        Path(yaml_config.ukbiobank_masks_dir) / pseudo_labels_dir_name,
        pseudo_labels_dir_name,
    )


def run_student_st(
    n_teacher_samples: str,
    pseudo_labels_dir_name: str,
    teacher_checkpoint: str,
    limit: Optional[int] = None,
):
    student_experiment_subdir_name = (
        f"offlineST_student_{n_teacher_samples}_samples_{dir_suffix}"
    )
    tags = f'["OfflineST","StudentTraining","VesselSeg","PseudoLabels", "FullModel", "{n_teacher_samples}_samples"]'
    cmd = f"python run.py --experiment_id=uk_biobank_experiment --sam_model=vit_b --learning_rate=0.0003 --batch_size=16 --epochs=5 --weight_decay=1e-4 --early_stopping_patience=3 --visualize_n_segmentations=5 --gamma=0.85 --step_size=1 --best_model_metric=IoU --minimize_best_model_metric=false --from_checkpoint={teacher_checkpoint} --image_encoder_lr=0.00005 --prompt_encoder_lr=0.0003 --mask_decoder_lr=0.0001 --use_wandb=true --drive_test_equals_val=false --amp=true --wandb_experiment_name={debug_prefix}ukbiobank_offlineST_teacher{n_teacher_samples}samples --augment_train=false --pseudo_labels_dir={pseudo_labels_dir_name} --results_subdir_name={student_experiment_subdir_name}"
    if limit != None:
        cmd += f" --limit={limit}"
    cmd += f" --wandb_tags={shlex.quote(tags)}"
    exec(cmd)
    student_checkpoint_dir = (
        Path(yaml_config.results_dir)
        / "uk_biobank_experiment"
        / student_experiment_subdir_name
    )
    student_dirs = os.listdir(student_checkpoint_dir)
    assert (
        len(student_dirs) == 1
    ), f"Expected 1 dir in {student_checkpoint_dir}, got {len(student_dirs)}"
    return student_checkpoint_dir / student_dirs[0] / "model.pt"


def run_full_fine_tuning(
    prefix: Literal["NoST", "OfflineST"], n_teacher_samples: str, checkpoint_path: str
):
    subdir_name = f"{prefix}_student_fft_{n_teacher_samples}_samples_{dir_suffix}"
    cmd = f"python run.py --experiment_id=multi_ds_vessel --sam_model=vit_b --learning_rate=0.0003 --batch_size=3 --epochs=100 --weight_decay=1e-4 --early_stopping_patience=4 --visualize_n_segmentations=5 --gamma=0.85 --step_size=1 --best_model_metric=IoU --minimize_best_model_metric=false --from_checkpoint={checkpoint_path} --image_encoder_lr=0.00005 --prompt_encoder_lr=0.0003 --mask_decoder_lr=0.0001 --use_wandb=true --drive_test_equals_val=false --amp=true --wandb_experiment_name={debug_prefix}vessels_gt_{prefix}_fft_{n_teacher_samples}_samples --results_subdir_name={subdir_name}"
    if n_teacher_samples != "all":
        cmd += f" --limit_train_samples={n_teacher_samples}"

    tags = [
        "OfflineST",
        "FinalFinetuning",
        "VesselSeg",
        "GT",
        "FullModel",
        f"{n_teacher_samples}_samples",
    ]
    if prefix == "NoST":
        tags.append("Baseline")
    cmd += f" --wandb_tags={shlex.quote(json.dumps(tags))}"
    exec(cmd)
    checkpoint_dir = (
        Path(yaml_config.results_dir) / "multi_ds_vessel_experiment" / subdir_name
    )
    dirs = os.listdir(checkpoint_dir)
    assert len(dirs) == 1, f"Expected 1 dir in {checkpoint_dir}, got {len(dirs)}"
    return checkpoint_dir / dirs[0] / "model.pt"


if __name__ == "__main__":
    args = parse_args()
    debug = args.debug
    debug_prefix = "DEBUG_" if debug else ""
    n_teacher_samples = str(args.n_teacher_samples) if args.n_teacher_samples else "all"

    teacher_checkpoint = run_teacher_pretraining1(n_teacher_samples)
    teacher_checkpoint = run_teacher_pretraining2(
        n_teacher_samples, str(teacher_checkpoint)
    )
    print(
        f"Teacher trained and saved in {teacher_checkpoint}. Generating pseudo labels..."
    )
    pseudo_labels_dir, pseudo_labels_dir_name = generate_pseudo_labels(
        n_teacher_samples, str(teacher_checkpoint), limit=1000 if debug else None
    )
    print(f"Generated pseudo labels in {pseudo_labels_dir}")

    print("Training student...")
    student_checkpoint = run_student_st(
        n_teacher_samples,
        pseudo_labels_dir_name,
        str(teacher_checkpoint),
        limit=1000 if debug else None,
    )

    print("Training FFT baseline")
    run_full_fine_tuning("NoST", n_teacher_samples, str(teacher_checkpoint))
    print("Training FFT offlineST")
    run_full_fine_tuning("OfflineST", n_teacher_samples, str(student_checkpoint))
