import os
from typing import Literal, Optional
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
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        help="Path to checkpoint of teacher model",
    )
    parser.add_argument(
        "--skip_pseudo_label_gen",
        action="store_true",
        help="Skip pseudo label generation. If set, the pseudo labels must have been generated already.",
    )
    parser.add_argument(
        "--gt_patches",
        type=int,
        help="Number of patches per original sample to use for ground truth fine-tuning",
        default=4,
    )
    parser.add_argument(
        "--freeze_image_encoder_in_st",
        action="store_true",
        help="Freeze image encoder in self training",
    )
    args = parser.parse_args()
    return args


def exec(cmd: str):
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)

    if result.returncode == 0:
        print("Command Output:\n", result.stdout)
    else:
        raise Exception(f"Error while executing '{cmd}':\n", result.stderr)


def generate_pseudo_labels(
    n_teacher_samples: str,
    teacher_checkpoint: str,
    limit: Optional[int] = None,
    skip_pseudo_label_gen: bool = False,
):
    pseudo_labels_dir_name = f"teacher_{n_teacher_samples}_samples"

    if not skip_pseudo_label_gen:
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
    freeze_image_encoder: bool = False,
):
    #python run.py --experiment_id=offline_st --sam_model=vit_b --learning_rate=0.0003 --batch_size=16 --epochs=5 --weight_decay=1e-4 --early_stopping_patience=2 
    # --visualize_n_segmentations=5 --gamma=0.85 --step_size=1 --best_model_metric=IoU --minimize_best_model_metric=false 
    # --from_checkpoint=/dhc/groups/mp2024cl2/results/multi_ds_vessel_experiment/baseline_patched4_all_samples_aug/2025-02-16_01#38#17/model.pt 
    # --prompt_encoder_lr=0.0003 --mask_decoder_lr=0.0001 --use_wandb=true --amp=true --wandb_experiment_name=ukbiobank_offlineST_teacherallsamples_frozenImgEnc_mixedlabels 
    # --pseudo_labels_dir=teacher_all_samples --results_subdir_name=offlineST_student_all_samples_frozenImgEnc_mixedLabels 
    # --wandb_tags="[\"OfflineST\", \"StudentTraining\", \"VesselSeg\", \"PseudoLabelsMixedGt\", \"all_samples\", \"FrozenImageEncoder\"]"
    student_experiment_subdir_name = (
        f"offlineST_student_{n_teacher_samples}_samples_{dir_suffix}"
    )
    wandb_exp_name = (
        f"{debug_prefix}ukbiobank_offlineST_teacher{n_teacher_samples}samples"
    )
    tags = [
        "OfflineST",
        "StudentTraining",
        "VesselSeg",
        "PseudoLabels",
        "FullModel",
        f"{n_teacher_samples}_samples",
    ]
    if freeze_image_encoder:
        student_experiment_subdir_name += "_frozenImgEnc"
        wandb_exp_name += "_frozenImgEnc"
        tags.append("FrozenImageEncoder")
    cmd = f"python run.py --experiment_id=uk_biobank_experiment --sam_model=vit_b --learning_rate=0.0003 --batch_size=16 --epochs=5 --weight_decay=1e-4 --early_stopping_patience=3 --visualize_n_segmentations=5 --gamma=0.85 --step_size=1 --best_model_metric=IoU --minimize_best_model_metric=false --from_checkpoint={teacher_checkpoint} --prompt_encoder_lr=0.0003 --mask_decoder_lr=0.0001 --use_wandb=true --drive_test_equals_val=false --amp=true --wandb_experiment_name={wandb_exp_name} --augment_train=false --pseudo_labels_dir={pseudo_labels_dir_name} --results_subdir_name={student_experiment_subdir_name}"
    if limit != None:
        cmd += f" --limit={limit}"

    if not freeze_image_encoder:
        cmd += " --image_encoder_lr=0.00005"
    cmd += f" --wandb_tags={shlex.quote(json.dumps(tags))}"
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
    prefix: Literal["NoST", "OfflineST"],
    n_teacher_samples: str,
    checkpoint_path: str,
    patches: Optional[int] = None,
    offlineST_frozenImgEnc: bool = False,
):
    patching_name = f"_patched{patches}" if patches != None else ""
    subdir_name = (
        f"{prefix}{patching_name}_student_fft_{n_teacher_samples}_samples_{dir_suffix}"
    )
    if offlineST_frozenImgEnc:
        subdir_name += "_frozenImgEnc"
    cmd = f"python run.py --experiment_id=multi_ds_vessel --sam_model=vit_b --learning_rate=0.0003 --batch_size=3 --epochs=100 --weight_decay=1e-4 --early_stopping_patience=5 --visualize_n_segmentations=5 --gamma=0.95 --step_size=3 --best_model_metric=IoU --minimize_best_model_metric=false --from_checkpoint={checkpoint_path} --image_encoder_lr=0.00001 --prompt_encoder_lr=0.0001 --mask_decoder_lr=0.0001 --use_wandb=true --drive_test_equals_val=false --amp=true --wandb_experiment_name={debug_prefix}vessels_gt{patching_name}_{prefix}_fft_{n_teacher_samples}_samples --results_subdir_name={subdir_name}"
    if n_teacher_samples != "all":
        cmd += f" --limit_train_samples={n_teacher_samples * (patches if patches != None else 1)}"

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
    if patches != None:
        tags.append(f"Patched{patches}")
        tags.append("Patched")
        cmd += f" --patch_samples={patches}"
    if offlineST_frozenImgEnc:
        tags.append("FrozenImageEncoder")

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

    teacher_checkpoint = args.teacher_checkpoint
    print(
        f"Teacher trained and saved in {teacher_checkpoint}. Generating pseudo labels..."
    )
    pseudo_labels_dir, pseudo_labels_dir_name = generate_pseudo_labels(
        n_teacher_samples,
        str(teacher_checkpoint),
        limit=1000 if debug else None,
        skip_pseudo_label_gen=args.skip_pseudo_label_gen,
    )
    print(f"Generated pseudo labels in {pseudo_labels_dir}")

    print("Training student...")
    student_checkpoint = run_student_st(
        n_teacher_samples,
        pseudo_labels_dir_name,
        str(teacher_checkpoint),
        limit=1000 if debug else None,
        freeze_image_encoder=args.freeze_image_encoder_in_st,
    )

    print("Training FFT offlineST")
    run_full_fine_tuning(
        "OfflineST",
        n_teacher_samples,
        str(student_checkpoint),
        patches=args.gt_patches,
        offlineST_frozenImgEnc=args.freeze_image_encoder_in_st,
    )
