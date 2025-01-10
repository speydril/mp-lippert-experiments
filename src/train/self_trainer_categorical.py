from math import floor
from typing import cast
import torch
from torch.utils.data import DataLoader
from src.models.base_model import ModelOutput
from src.datasets.base_dataset import Batch
from src.train.self_trainer import SelfTrainer


class SelfTrainerCategorical(SelfTrainer):
    def _preprocess_labels(self, batch: Batch):
        logits = self.model.forward(batch).logits
        one_hot = torch.zeros_like(logits)
        indices = logits.argmax(dim=1)
        one_hot[torch.arange(logits.size(0)), indices] = 1
        return one_hot

    def _augment_batch(self, batch: Batch) -> Batch:
        return self._gaussian_noise_augmentation(batch)
