from typing import Literal, Any, Optional
import torch
from torch.optim.optimizer import Optimizer
from src.datasets.offline_st_dataset import OfflineSTTrainDataset, OfflineStDatasetArgs
from src.datasets.joined_retina_dataset import (
    JoinedRetinaDataset,
    JoinedRetinaDatasetArgs,
)
from src.models.auto_sam_model import AutoSamModel, AutoSamModelArgs
from src.experiments.base_experiment import BaseExperiment, BaseExperimentArgs
from src.models.base_model import BaseModel
from src.args.yaml_config import YamlConfigModel
from src.optimizers.adam import AdamArgs
from src.schedulers.step_lr import StepLRArgs, create_steplr_scheduler
from typing import cast
import os
from pydantic import Field


class OfflineSTExperimentArgs(
    BaseExperimentArgs, AdamArgs, StepLRArgs, AutoSamModelArgs, OfflineStDatasetArgs
):
    prompt_encoder_checkpoint: Optional[str] = Field(
        default=None, description="Path to prompt encoder checkpoint"
    )
    visualize_n_segmentations: int = Field(
        default=3, description="Number of images of test set to segment and visualize"
    )
    image_encoder_lr: Optional[float] = Field(
        default=None, description="Learning rate for image encoder"
    )
    mask_decoder_lr: Optional[float] = Field(
        default=None, description="Learning rate for mask decoder"
    )
    prompt_encoder_lr: Optional[float] = Field(
        default=None, description="Learning rate for prompt encoder"
    )


class OfflineSTExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = OfflineSTExperimentArgs(**config)

        gt_config = JoinedRetinaDatasetArgs(drive_test_equals_val=False)
        self.joined_retina = JoinedRetinaDataset.from_config(
            gt_config, yaml_config, self.config.seed
        )
        self.train_ds = OfflineSTTrainDataset(
            self.config, yaml_config, self.config.seed
        )

        super().__init__(config, yaml_config)

    def get_name(self) -> str:
        return "offline_st_experiment"

    def _create_dataset(self, split: Literal["train", "val", "test"] = "train"):
        if split == "train":
            return self.train_ds
        else:
            return self.joined_retina.get_split(split)

    def _create_model(self) -> BaseModel:
        model = AutoSamModel(self.config, image_encoder_no_grad=False)
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
        return OfflineSTExperimentArgs

    def create_optimizer(self) -> Optimizer:
        params = [
            {
                "params": cast(AutoSamModel, self.model).sam.mask_decoder.parameters(),
                "lr": (
                    self.config.mask_decoder_lr
                    if self.config.mask_decoder_lr is not None
                    else self.config.learning_rate
                ),
            },
            {
                "params": cast(
                    AutoSamModel, self.model
                ).sam.prompt_encoder.parameters(),
                "lr": (
                    self.config.prompt_encoder_lr
                    if self.config.prompt_encoder_lr is not None
                    else self.config.learning_rate
                ),
            },
        ]
        if self.config.image_encoder_lr is not None:
            params.append(
                {
                    "params": cast(
                        AutoSamModel, self.model
                    ).sam.image_encoder.parameters(),
                    "lr": (self.config.image_encoder_lr),
                },
            )
        else:
            print("Image encoder lr is none, therefore parameters are not trainable")

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

    def run_after_training(self, trained_model: BaseModel):
        model = cast(AutoSamModel, trained_model)

        def predict_visualize(split: Literal["train", "test"]):
            out_dir = os.path.join(self.results_dir, f"{split}_visualizations")
            os.makedirs(out_dir, exist_ok=True)
            ds = self._create_dataset(split)
            print(
                f"\nCreating {self.config.visualize_n_segmentations} {split} segmentations"
            )

            file_refs = ds.get_file_refs()
            for i in range(min(len(file_refs), self.config.visualize_n_segmentations)):
                sample = file_refs[i]
                out_path = os.path.join(out_dir, f"{i}.png")
                model.segment_and_write_image_from_file(
                    str(sample.img_path),
                    out_path,
                    gts_path=(
                        str(sample.gt_path) if sample.gt_path is not None else None
                    ),
                )
                print(
                    f"{i+1}/{self.config.visualize_n_segmentations} {split} segmentations created\r",
                    end="",
                )

        predict_visualize("train")
        predict_visualize("test")
