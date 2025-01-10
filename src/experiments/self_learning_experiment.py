from math import floor
from typing import Literal, Any, Optional, Type
from click import prompt
import torch
from torch.optim.optimizer import Optimizer
from src.datasets.joined_retina_dataset import (
    JoinedRetinaDataset,
    JoinedRetinaDatasetArgs,
)
from torch.utils.data import DataLoader
from src.datasets.ukbiobank_dataset import UkBiobankDataset, UkBiobankDatasetArgs
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


class SelfLearningExperimentArgs(
    BaseExperimentArgs,
    AdamArgs,
    StepLRArgs,
    AutoSamModelArgs,
    UkBiobankDatasetArgs,
    JoinedRetinaDatasetArgs,
):
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
    minimum_student_ratio: float = Field(
        default=0.1, description="Exponential moving average decay"
    )
    maximum_student_ratio: float = Field(
        default=0.5, description="Exponential moving average decay"
    )


class SelfLearningExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = SelfLearningExperimentArgs(**config)
        super().__init__(config, yaml_config)

        # Setting up student model
        # Same architecture, same initial checkpoint
        self.teacher_model = self._create_model()
        if self.base_config.from_checkpoint is not None:
            self.teacher_model.load_state_dict(
                torch.load(self.base_config.from_checkpoint, map_location="cuda"),
                strict=True,
            )
        self.teacher_model.to("cuda")

    def _init_super(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        super().__init__(config, yaml_config)

    def get_name(self) -> str:
        return "self_learning_experiment"

    def _create_trainer(self):
        from src.train.self_trainer import SelfTrainer

        return SelfTrainer(self)

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        if split == "train":
            return UkBiobankDataset(
                config=self.config,
                yaml_config=self.yaml_config,
                with_masks=False,
            )
        else:
            return JoinedRetinaDataset.from_config(
                self.config, self.yaml_config
            ).get_split(split)

    def _create_model(self) -> BaseModel:
        model = AutoSamModel(self.config, image_encoder_no_grad=False)
        return model

    @classmethod
    def get_args_model(cls) -> Type[BaseExperimentArgs]:
        return SelfLearningExperimentArgs

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            return [
                {
                    "params": cast(
                        AutoSamModel, self.model
                    ).sam.image_encoder.parameters(),
                    "lr": (
                        self.config.image_encoder_lr
                        if self.config.image_encoder_lr is not None
                        else self.config.learning_rate
                    ),
                },
                {
                    "params": cast(
                        AutoSamModel, self.model
                    ).sam.mask_decoder.parameters(),
                    "lr": (
                        self.config.mask_decoder_lr
                        if self.config.mask_decoder_lr is not None
                        else self.config.learning_rate
                    ),
                },
                {
                    "params": cast(
                        AutoSamModel, self.model
                    ).prompt_encoder.parameters(),
                    "lr": (
                        self.config.prompt_encoder_lr
                        if self.config.prompt_encoder_lr is not None
                        else self.config.learning_rate
                    ),
                },
            ]

        return torch.optim.Adam(
            get_trainable_params(),
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

    def run(self):
        # Execute finetuning
        super().run()
