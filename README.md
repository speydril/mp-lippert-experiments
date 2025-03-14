# Experiments of Master's Project at Lippert Chair
This repository contains the code implemented in the context of the Master's Project of the Chair of Prof. Lippert 2024/25 at Hasso Plattner Institute. 
This is the implementation for the report "Is SAM a Good Teacher? Self-Training With SAM for Retinal Vessel Segmentation".

## Getting started
1. Clone this repository
2. Navigate into the cloned repository in terminal via `cd`
3. Install conda env via `conda env create -f environment.yaml`
4. Activate conda env `conda activate autosam`

## Reproducing our experiments
We executed our experiments on a system with 32GB of RAM and a Nvidia A40.
### Baseline/Teacher training
Example for training on 8 GT samples: `python run.py --experiment_id=multi_ds_vessel --sam_model=vit_b --learning_rate=0.0003 --prompt_encoder_lr=0.0001 --image_encoder_lr=0.00001 --mask_decoder_lr=0.0001 --batch_size=3 --epochs=50 --weight_decay=1e-4 --early_stopping_patience=5 --visualize_n_segmentations=5 --gamma=0.95 --step_size=3 --best_model_metric=IoU --minimize_best_model_metric=false --drive_test_equals_val=false --use_wandb=true --amp=true --results_subdir_name=baseline_patched4_8_samples_aug --wandb_experiment_name=gt_vessels_patched4_baseline_8_samples_aug --wandb_tags='["StrictPatching","Baseline","VesselSeg","GT","FullModel", "8_samples", "Patched","Patched4", "AugmentTrain"]' --augment_train=true --limit_train_samples=32 --patch_samples=4`.
Note that `limit_train_samples` is set to `32` here. This is because we have 4 patches per original sample and thereby get `4*8=32` training samples. However, the original samples from which the patches are sampled are still only 8. We have double checked this in [prevent_leakage.ipynb](./src/wild-west/prevent_leakage.ipynb).
### Offline Self-Training
For ease of use, we have added a script that executes the steps of OfflineST consecutively without manual intervention.
In can be executed via `python src/wild-west/offline_st_run.py --teacher_checkpoint=$1 --n_teacher_samples=$2 --gt_ratio=$3 --gt_patches=4 --skip_pseudo_label_gen`. 
The variables to be replaced are: 
- `$1`: checkpoint of the teacher model
- `$2`: n_teacher_samples, i.e. the number of original samples the teacher was limited to during training. E.g. if the teacher is limited to 8 GT samples, this should be 8 (even when patching)
- `$3`: gt_ratio, i.e. the average ratio that GT samples should have in the Self-Training step. The GT samples are sampled repeatedly from the same set of GT samples. E.g. if GT samples are limited to 8 and gt_ratio is 0.25, the training data will contain on average 25% GT samples, sampled repeatedly from the same set of 8.
Note that `--skip_pseudo_label_gen` is optional and should only be added when the script has been executed once for the specific n_teacher_samples.

### Online Self-Training

