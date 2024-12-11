from dataclasses import dataclass
from math import floor
import os
from pathlib import Path
from typing import Literal, Optional
import numpy as np
from sympy import im
import torch
import cv2
from typing_extensions import Self
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset, Batch, Sample
from pydantic import BaseModel

from src.datasets.refuge_dataset import get_polyp_transform
from src.models.segment_anything.utils.transforms import ResizeLongestSide
from src.util.image_util import calculate_rgb_mean_std


class UkBiobankDatasetArgs(BaseModel):
    train_percentage: float = 0.8
    val_percentage: float = 0.15
    test_percentage: float = 0.05
    filter_scores_filepath: str = (
        "/dhc/groups/mp2024cl2/ukbiobank_filters/filter_predictions.csv"
    )
    mask_iteration: int = 0


@dataclass
class BiobankSampleReference:
    img_path: Path
    gt_path: Path | None
    split: str


@dataclass
class BiobankSample(Sample):
    split: str
    original_size: torch.Tensor
    image_size: torch.Tensor
    img_path: Path


@dataclass
class BiobankBatch(Batch):
    original_size: torch.Tensor
    image_size: torch.Tensor
    file_paths: list[Path]


class UkBiobankDataset(BaseDataset):

    def __init__(
        self,
        config: UkBiobankDatasetArgs,
        yaml_config: YamlConfigModel,
        samples: Optional[list[BiobankSampleReference]] = None,
        image_enc_img_size=1024,
        with_masks=False,
        random_augmentation_for_all_splits=False,
    ):
        self.config = config
        self.yaml_config = yaml_config
        self.with_masks = with_masks
        self.samples = self.load_data() if samples is None else samples
        pixel_mean, pixel_std = calculate_rgb_mean_std(
            [str(s.img_path) for s in self.samples],
            os.path.join(yaml_config.cache_dir, "ukbiobank_mean_std.pkl"),
        )
        self.sam_trans = ResizeLongestSide(
            image_enc_img_size, pixel_mean=pixel_mean, pixel_std=pixel_std
        )
        self.random_augmentation_for_all_splits = random_augmentation_for_all_splits

    def __getitem__(self, index: int) -> BiobankSample:
        sample = self.samples[index]
        train_transform, test_transform = get_polyp_transform()

        augmentations = (
            test_transform
            if sample.split == "test" and not self.random_augmentation_for_all_splits
            else train_transform
        )
        image = self.cv2_loader(str(sample.img_path), is_mask=False)
        gt = (
            self.cv2_loader(str(sample.gt_path), is_mask=True)
            if self.with_masks
            else np.zeros_like(image)
        )

        img, mask = augmentations(image, gt)

        mask = self.sam_trans.apply_image_torch(torch.Tensor(mask))
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        original_size = tuple(img.shape[1:3])
        img = self.sam_trans.apply_image_torch(torch.Tensor(img))
        image_size = tuple(img.shape[1:3])

        return BiobankSample(
            input=self.sam_trans.preprocess(img),
            target=mask,
            original_size=torch.Tensor(original_size),
            image_size=torch.Tensor(image_size),
            split=sample.split,
            img_path=sample.img_path,
        )

    def __len__(self):
        return len(self.samples)

    def get_collate_fn(self):  # type: ignore
        def collate(samples: list[BiobankSample]):
            inputs = torch.stack([s.input for s in samples])
            targets = torch.stack([s.target for s in samples])
            original_size = torch.stack([s.original_size for s in samples])
            image_size = torch.stack([s.image_size for s in samples])
            return BiobankBatch(
                inputs,
                targets,
                original_size=original_size,
                image_size=image_size,
                file_paths=[s.img_path for s in samples],
            )

        return collate

    def get_split(self, split: Literal["train", "val", "test"]) -> Self:
        return self.__class__(
            self.config,
            self.yaml_config,
            [sample for sample in self.samples if sample.split == split],
            with_masks=self.with_masks,
        )

    def load_data(self) -> list[BiobankSampleReference]:
        mask_folder = (
            Path(self.yaml_config.ukbiobank_masks_dir)
            / f"v{self.config.mask_iteration}"
        )
        filter_scores_filepath = Path(self.config.filter_scores_filepath)

        selected_samples = []
        with open(filter_scores_filepath, "r") as f:
            filter_scores = f.readlines()
            for line in filter_scores[1:]:
                path, neg_prob, pos_prob, prediction = line.strip().split(",")
                if prediction == "0":
                    continue
                selected_samples.append(path)

        sample_paths = [
            (path, mask_folder / path.split("/")[-1] if self.with_masks else None)
            for path in selected_samples
            if path.endswith(".png")
        ]

        train = self.load_data_for_split("train", sample_paths)
        val = self.load_data_for_split("val", sample_paths)
        test = self.load_data_for_split("test", sample_paths)

        return train + val + test

    def load_data_for_split(
        self, split, sample_paths: list[tuple[Path, Path | None]]
    ) -> list[BiobankSampleReference]:
        index_offset = (
            0
            if split == "train"
            else (
                floor(len(sample_paths) * self.config.train_percentage)
                if split == "val"
                else floor(
                    len(sample_paths)
                    * (self.config.train_percentage + self.config.val_percentage)
                )
            )
        )
        length = (
            floor(len(sample_paths) * self.config.train_percentage)
            if split == "train"
            else (
                floor(len(sample_paths) * self.config.val_percentage)
                if split == "val"
                else floor(len(sample_paths) * self.config.test_percentage)
            )
        )

        return [
            BiobankSampleReference(img_path=img_path, gt_path=gt_path, split=split)
            for img_path, gt_path in sample_paths[index_offset : index_offset + length]
        ]

    def cv2_loader(self, path: str, is_mask: bool):
        if is_mask:
            img = cv2.imread(path, 0)
            img[img > 0] = 1
        else:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img
