from typing import Literal, Any, Optional
import torch
from torch.optim.optimizer import Optimizer
from src.datasets.joined_patched_retina_dataset import JoinedPatchedRetinaDataset
from src.datasets.joined_retina_dataset import (
    JoinedRetinaDataset,
    JoinedRetinaDatasetArgs,
)
from src.models.auto_sam_model import AutoSamModel, AutoSamModelArgs
from src.experiments.base_experiment import BaseExperiment, BaseExperimentArgs
from src.models.base_model import BaseModel
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset
from src.optimizers.adam import AdamArgs
from src.schedulers.step_lr import StepLRArgs, create_steplr_scheduler
from typing import cast
import os
from pydantic import Field
from src.util.eval_util import evaluate_model


class MultiDSVesselExperimentArgs(
    BaseExperimentArgs,
    AdamArgs,
    StepLRArgs,
    AutoSamModelArgs,
    JoinedRetinaDatasetArgs,
):
    prompt_encoder_checkpoint: Optional[str] = Field(
        default=None, description="Path to prompt encoder checkpoint"
    )
    visualize_n_segmentations: int = Field(
        default=3, description="Number of images of test set to segment and visualize"
    )
    image_encoder_lr: Optional[float] = Field(
        default=None,
        description="Learning rate for image encoder, if None image encoder is frozen",
    )
    mask_decoder_lr: Optional[float] = Field(
        default=None,
        description="Learning rate for mask decoder,  if None image encoder is frozen",
    )
    prompt_encoder_lr: Optional[float] = Field(
        default=None,
        description="Learning rate for prompt encoder, if None, general learning rate is used",
    )
    limit_train_samples: Optional[int] = Field(
        default=None,
        description="Limit number of training samples, i.e. image mask pairs to include (if patch_samples is True, this is the number of patches)",
    )
    patch_samples: Optional[Literal[4, 16]] = Field(
        default=None, description="Patch samples into 4 or 16 parts"
    )


class MultiDsVesselExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = MultiDSVesselExperimentArgs(**config)

        self.ds = (
            JoinedPatchedRetinaDataset.from_config(
                self.config,
                yaml_config,
                self.config.seed,
                patches=self.config.patch_samples,
            )
            if self.config.patch_samples is not None
            else JoinedRetinaDataset.from_config(
                self.config, yaml_config, self.config.seed
            )
        )
        super().__init__(config, yaml_config)

    def get_name(self) -> str:
        return "multi_ds_vessel_experiment"

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        if split == "train":
            return self.ds.get_split(
                split, limit_samples=self.config.limit_train_samples
            )
        return self.ds.get_split(split)

    def _create_model(self) -> BaseModel:
        image_encoder_no_grad = self.config.image_encoder_lr is None
        model = AutoSamModel(self.config, image_encoder_no_grad)

        if self.config.prompt_encoder_checkpoint is not None:
            print(
                f"loading prompt-encoder model from checkpoint {self.config.prompt_encoder_checkpoint}"
            )
            model.prompt_encoder.load_state_dict(
                torch.load(self.config.prompt_encoder_checkpoint, map_location="cuda"),
                strict=True,
            )
        return model

    @classmethod
    def get_args_model(cls):
        return MultiDSVesselExperimentArgs

    def create_optimizer(self) -> Optimizer:
        prompt_enc_params: dict = {
            "params": cast(AutoSamModel, self.model).prompt_encoder.parameters(),
        }
        if self.config.prompt_encoder_lr is not None:
            prompt_enc_params["lr"] = self.config.prompt_encoder_lr

        params = [prompt_enc_params]

        if self.config.image_encoder_lr is not None:
            params.append(
                {
                    "params": cast(
                        AutoSamModel, self.model
                    ).sam.image_encoder.parameters(),
                    "lr": self.config.image_encoder_lr,
                }
            )

        # Always add mask decoder to optimizer to allow for Automatic Mixed Precision to work even when mask decoder isn't trained
        # See bottom of https://chatgpt.com/share/675ae2c8-fff4-800c-8a5b-cecc352df76a

        mask_decoder = cast(AutoSamModel, self.model).sam.mask_decoder
        params.append(
            {
                "params": mask_decoder.parameters(),
                "lr": (
                    self.config.mask_decoder_lr
                    if self.config.mask_decoder_lr is not None
                    else 0
                ),
            }
        )

        return torch.optim.Adam(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=self.config.eps,
        )

    def create_scheduler(
        self, optimizer: Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:

        return create_steplr_scheduler(optimizer, self.config)

    def get_loss_name(self) -> str:
        return "dice+bce"

    def store_trained_model(self, trained_model: torch.nn.Module):
        model = cast(AutoSamModel, trained_model)
        torch.save(
            model.prompt_encoder.state_dict(),
            os.path.join(self.results_dir, "prompt_encoder.pt"),
        )
        torch.save(
            model.state_dict(),
            os.path.join(self.results_dir, "model.pt"),
        )

    def run_after_training(self, trained_model: BaseModel):
        model = cast(AutoSamModel, trained_model)
        val_metrics = evaluate_model(trained_model, self._create_dataloader("val"), 5)

        def predict_visualize(split: Literal["train", "test"]):
            out_dir = os.path.join(self.results_dir, f"{split}_visualizations")
            os.makedirs(out_dir, exist_ok=True)
            iou_threshold_dir = os.path.join(out_dir, "iou_threshold")
            auc_threshold_dir = os.path.join(out_dir, "auc_threshold")
            os.makedirs(iou_threshold_dir, exist_ok=True)
            os.makedirs(auc_threshold_dir, exist_ok=True)

            ds = self.ds.get_split(split)
            print(
                f"\nCreating {self.config.visualize_n_segmentations} {split} segmentations"
            )
            for i in range(min(len(ds), self.config.visualize_n_segmentations)):
                sample = ds.get_file_refs()[i]
                out_path = os.path.join(out_dir, f"{i}.png")
                iou_threshold_out_path = os.path.join(iou_threshold_dir, f"{i}.png")
                auc_threshold_out_path = os.path.join(auc_threshold_dir, f"{i}.png")
                model.segment_and_write_image_from_file(
                    sample.img_path, out_path, gts_path=sample.gt_path
                )
                iou_threshold = val_metrics.get("iou_threshold", 0.5)
                model.segment_and_write_image_from_file(
                    sample.img_path,
                    iou_threshold_out_path,
                    gts_path=sample.gt_path,
                    threshold=iou_threshold,
                )
                auc_threshold = val_metrics.get("auc_threshold", 0.5)
                model.segment_and_write_image_from_file(
                    sample.img_path,
                    auc_threshold_out_path,
                    gts_path=sample.gt_path,
                    threshold=auc_threshold,
                )

                if self.config.patch_samples:
                    patched_out_dir = os.path.join(
                        out_dir, f"{self.config.patch_samples}patched"
                    )
                    patched_out_path = os.path.join(patched_out_dir, f"{i}.png")
                    os.makedirs(patched_out_dir, exist_ok=True)
                    model.segment_and_write_image_from_file(
                        sample.img_path,
                        patched_out_path,
                        gts_path=sample.gt_path,
                        patches=(
                            self.config.patch_samples
                            if self.config.patch_samples
                            else None
                        ),
                    )
                    # IOU threshold optimized
                    patched_iou_dir = os.path.join(
                        iou_threshold_dir, f"{self.config.patch_samples}patched"
                    )
                    patched_iou_out = os.path.join(patched_iou_dir, f"{i}.png")
                    os.makedirs(patched_iou_dir, exist_ok=True)
                    model.segment_and_write_image_from_file(
                        sample.img_path,
                        patched_iou_out,
                        gts_path=sample.gt_path,
                        threshold=iou_threshold,
                        patches=(
                            self.config.patch_samples
                            if self.config.patch_samples
                            else None
                        ),
                    )
                    # AUC threshold optimized
                    patched_auc_dir = os.path.join(
                        auc_threshold_dir, f"{self.config.patch_samples}patched"
                    )
                    patched_auc_out = os.path.join(patched_auc_dir, f"{i}.png")
                    os.makedirs(patched_auc_dir, exist_ok=True)
                    model.segment_and_write_image_from_file(
                        sample.img_path,
                        patched_auc_out,
                        threshold=auc_threshold,
                        gts_path=sample.gt_path,
                        patches=(
                            self.config.patch_samples
                            if self.config.patch_samples
                            else None
                        ),
                    )
                print(
                    f"{i+1}/{self.config.visualize_n_segmentations} {split} segmentations created\r",
                    end="",
                )

        predict_visualize("train")
        predict_visualize("test")
