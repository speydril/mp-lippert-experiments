from typing import Literal, Any
import torch
from torch.optim.optimizer import Optimizer
from src.schedulers.step_lr import StepLRArgs
from src.experiments.self_learning_experiment import (
    SelfLearningExperiment,
)
from src.datasets.mnist_dataset import MnistDataset, MnistDatasetArgs
from src.models.mnist_fc_model import MnistFcModel, MnistFcModelArgs
from src.experiments.base_experiment import BaseExperimentArgs
from src.models.base_model import BaseModel
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset
from src.optimizers.adam import AdamArgs, create_adam_optimizer
from pydantic import Field


class SelfLearningExperimentDebugArgs(
    BaseExperimentArgs, AdamArgs, StepLRArgs, MnistFcModelArgs, MnistDatasetArgs
):
    labeled_limit: int = Field(
        default=1000, description="Number of labeled samples to use"
    )
    ema_decay: float = Field(
        default=0.999, description="Exponential moving average decay"
    )
    constant_ema_decay: bool = Field(default=True, description="Use constant EMA decay")


class SelfLearningExperimentDebug(SelfLearningExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        # Setting up labeled dataset and unlabeled dataset
        self.config = SelfLearningExperimentDebugArgs(**config)
        self._init_super(config, yaml_config)
        # Setting up student model
        # Same architecture, same initial checkpoint
        self.teacher_model = self._create_model()
        if self.base_config.from_checkpoint is not None:
            self.teacher_model.load_state_dict(
                torch.load(self.base_config.from_checkpoint, map_location="cuda"),
                strict=True,
            )
        self.teacher_model.to("cuda")

    def get_name(self) -> str:
        return "self_learning_experiment_debug"

    def _create_trainer(self):
        from src.train.self_trainer_categorical import SelfTrainerCategorical

        return SelfTrainerCategorical(self)

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        ds = MnistDataset(self.config, self.yaml_config, split=split)

        if split == "train":
            ds = MnistDataset(
                self.config,
                self.yaml_config,
                samples=ds.samples[self.config.labeled_limit :],
            )

        return ds

    def _create_model(self) -> BaseModel:
        model = MnistFcModel(self.config)
        return model

    @classmethod
    def get_args_model(cls):
        return SelfLearningExperimentDebugArgs

    def create_optimizer(self) -> Optimizer:
        return create_adam_optimizer(self.model, self.config)

    def get_loss_name(self) -> str:
        return "cross_entropy"
