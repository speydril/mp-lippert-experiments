from src.datasets.base_dataset import BaseDataset, Sample
from torchvision.datasets import MNIST
from pydantic import BaseModel
from src.args.yaml_config import YamlConfigModel
from typing import Literal, Optional
from math import floor
import torchvision.transforms as transforms
import torch
from typing_extensions import Self
import os


class MnistDatasetArgs(BaseModel):
    """Define arguments for the dataset here, i.e. preprocessing related stuff etc"""

    limit: Optional[int] = None
    pass


class MnistDataset(BaseDataset):
    def __init__(
        self,
        config: MnistDatasetArgs,
        yaml_config: YamlConfigModel,
        samples: Optional[list[Sample]] = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.yaml_config = yaml_config
        self.config = config
        self.samples = self.load_data(split) if samples is None else samples
        self.split = split

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def __len__(self):
        if self.config.limit and self.split == "train":
            return self.config.limit
        return len(self.samples)

    def load_data(self, split: Literal["train", "val", "test"]) -> list[Sample]:
        mnist_data = MNIST(
            os.path.join(self.yaml_config.cache_dir, "mnist"),
            train=split == "train",
            download=True,
        )
        image_transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Lambda(lambda x: x.view(-1)),  # Flatten the image to 1D
            ]
        )
        labels_matrix = torch.eye(10)
        target_transform = transforms.Lambda(lambda y: labels_matrix[y])
        samples = [
            Sample(image_transform(image), target_transform(target))
            for image, target in mnist_data
        ]

        if split == "train":
            return samples
        else:
            if split == "val":
                return samples[: floor(0.5 * len(samples))]
            else:
                return samples[floor(0.5 * len(samples)) :]
