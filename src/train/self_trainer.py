import os
from typing import cast

import numpy as np
import torch
import wandb


from src.models.auto_sam_model import norm_batch
from src.experiments.self_learning_experiment import SelfLearningExperiment
from src.datasets.base_dataset import Batch
from src.train.trainer import Trainer
from src.util.datatset_helper import generate_unsup_data


class SelfTrainer(Trainer):
    def __init__(self, experiment: SelfLearningExperiment):
        super().__init__(experiment)
        self.experiment = cast(SelfLearningExperiment, experiment)

        # Disable grad for teacher model
        for p in self.experiment.teacher_model.parameters():
            p.requires_grad = False

    def _get_eval_model(self):
        return self.experiment.teacher_model

    # Taken from https://github.com/usr922/FST/blob/main/semi_seg/train_semi.py
    # Line: 326
    def _augment_batch(self, batch: Batch) -> Batch:
        assert batch.target is not None

        # apply strong data augmentation: cutout or cutmix
        if np.random.uniform(0, 1) < 0.5:
            batch.input, batch.target = generate_unsup_data(batch.input, batch.target)

        return super()._augment_batch(batch)

    def _preprocess_labels(self, batch: Batch):
        self.experiment.teacher_model.eval()
        return norm_batch(self.experiment.teacher_model.forward(batch).logits)

    def _after_epoch_complete(self, epoch: int):
        # Adjust weights of teacher model
        with torch.no_grad():  # Disable gradient tracking
            linear = 1 - 1 / (epoch + 1)
            ema_decay = max(
                self.experiment.config.minimum_student_ratio,
                min(self.experiment.config.maximum_student_ratio, linear),
            )
            for t_params, s_params in zip(
                self.experiment.teacher_model.parameters(), self.model.parameters()
            ):
                t_params.data = (
                    ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                )
