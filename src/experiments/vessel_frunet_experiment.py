from typing import Literal, Any, Optional
import torch
from torch.optim.optimizer import Optimizer
from src.models.fr_unet_model import FRUnet, FRUnetArgs
from src.datasets.joined_retina_dataset import (
    JoinedRetinaDataset,
    JoinedRetinaDatasetArgs,
)
from src.experiments.base_experiment import BaseExperiment, BaseExperimentArgs
from src.models.base_model import BaseModel
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset
from src.optimizers.adam import AdamArgs
from src.schedulers.step_lr import StepLRArgs, create_steplr_scheduler
from typing import cast
import os
from pydantic import Field


class VesselFRUnetExperimentArgs(
    BaseExperimentArgs,
    AdamArgs,
    StepLRArgs,
    FRUnetArgs,
    JoinedRetinaDatasetArgs,
):
    visualize_n_segmentations: int = Field(
        default=3, description="Number of images of test set to segment and visualize"
    )
    limit_train_samples: Optional[int] = Field(
        default=None, description="Limit number of training samples"
    )


class VesselFRUnetExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = VesselFRUnetExperimentArgs(**config)

        self.ds = JoinedRetinaDataset.from_config(
            self.config, yaml_config, self.config.seed
        )
        super().__init__(config, yaml_config)

    def get_name(self) -> str:
        return "vessel_fr_unet"

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        if split == "train":
            self.ds.get_split(split, limit_samples=self.config.limit_train_samples)
        return self.ds.get_split(split)

    def _create_model(self) -> BaseModel:
        return FRUnet(self.config)

    @classmethod
    def get_args_model(cls):
        return VesselFRUnetExperimentArgs

    def create_optimizer(self) -> Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
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
        model = cast(FRUnet, trained_model)

        def predict_visualize(split: Literal["train", "test"]):
            out_dir = os.path.join(self.results_dir, f"{split}_visualizations")
            os.makedirs(out_dir, exist_ok=True)
            ds = self.ds.get_split(split)
            print(
                f"\nCreating {self.config.visualize_n_segmentations} {split} segmentations"
            )
            for i in range(min(len(ds), self.config.visualize_n_segmentations)):
                sample = ds.get_file_refs()[i]
                out_path = os.path.join(out_dir, f"{i}.png")
                model.segment_and_write_image_from_file(
                    sample.img_path, out_path, gts_path=sample.gt_path
                )
                print(
                    f"{i+1}/{self.config.visualize_n_segmentations} {split} segmentations created\r",
                    end="",
                )

        predict_visualize("train")
        predict_visualize("test")
