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
from pydantic import BaseModel, Field

from src.util.polyp_transform import get_polyp_transform
from src.models.segment_anything.utils.transforms import ResizeLongestSide


class UkBiobankDatasetArgs(BaseModel):
    train_percentage: float = 0.8
    val_percentage: float = 0.15
    test_percentage: float = 0.05
    filter_scores_filepath: str = (
        "/dhc/groups/mp2024cl2/ukbiobank_filters/filter_predictions.csv"
    )
    pseudo_labels_dir: Optional[str] = Field(
        default=None,
        description="Name of the the directory containing the pseudo labels to be filled into pattern <yaml_config.ukbiobank_masks_dir>/<pseudo_labels_dir>/generated_masks/[masks].png",
    )
    limit: Optional[int] = None



@dataclass
class BiobankSampleReference:
    img_path: Path
    gt_path: Path | None


@dataclass
class BiobankSample(Sample):
    original_size: torch.Tensor
    image_size: torch.Tensor
    img_path: Path
    gt_path: Path | None


@dataclass
class BiobankBatch(Batch):
    original_size: torch.Tensor
    image_size: torch.Tensor
    file_paths: list[Path]
    gt_paths: list[Path | None]


class UkBiobankDataset(BaseDataset):

    def __init__(
        self,
        config: UkBiobankDatasetArgs,
        yaml_config: YamlConfigModel,
        samples: Optional[list[BiobankSampleReference]] = None,
        with_masks=False,
        augment_inputs=False,
    ):
        self.config = config
        self.yaml_config = yaml_config
        self.with_masks = with_masks
        self.samples = self.load_data() if samples is None else samples
        self.samples = self.samples if config.limit is None else self.samples[:config.limit]
        pixel_mean, pixel_std = (
            self.yaml_config.fundus_pixel_mean,
            self.yaml_config.fundus_pixel_std,
        )
        self.sam_trans = ResizeLongestSide(
            self.yaml_config.fundus_resize_img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.augment_inputs = augment_inputs
        total_percentage = (
            self.config.train_percentage
            + self.config.val_percentage
            + self.config.test_percentage
        )
        assert (
            total_percentage <= 1.0
        ), f"train + val + test percantages > 1 (it is {total_percentage})"

    def get_file_refs(self) -> list[BiobankSampleReference]:
        return self.samples

    def __getitem__(self, index: int) -> BiobankSample:
        sample = self.samples[index]
        return self.get_sample_from_file(sample)

    def get_sample_from_file(self, file_ref: BiobankSampleReference):
        train_transform, test_transform = get_polyp_transform()

        augmentations = train_transform if self.augment_inputs else test_transform
        image = self.cv2_loader(str(file_ref.img_path), is_mask=False)
        gt = (
            self.cv2_loader(str(file_ref.gt_path), is_mask=True)
            if self.with_masks
            else np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
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
            target=self.sam_trans.preprocess(mask),
            original_size=torch.Tensor(original_size),
            image_size=torch.Tensor(image_size),
            img_path=file_ref.img_path,
            gt_path=file_ref.gt_path,
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
                gt_paths=[s.gt_path for s in samples],
            )

        return collate

    def get_split(self, split: Literal["train", "val", "test"]) -> Self:
        index_offset = (
            0
            if split == "train"
            else (
                floor(len(self.samples) * self.config.train_percentage)
                if split == "val"
                else floor(
                    len(self.samples)
                    * (self.config.train_percentage + self.config.val_percentage)
                )
            )
        )
        length = (
            floor(len(self.samples) * self.config.train_percentage)
            if split == "train"
            else (
                floor(len(self.samples) * self.config.val_percentage)
                if split == "val"
                else floor(len(self.samples) * self.config.test_percentage)
            )
        )

        return self.__class__(
            self.config,
            self.yaml_config,
            self.samples[index_offset : index_offset + length],
            with_masks=self.with_masks,
            augment_inputs=self.augment_inputs if split == "train" else False,
        )

    def load_data(self) -> list[BiobankSampleReference]:

        filter_scores_filepath = Path(self.config.filter_scores_filepath)

        selected_samples = []
        with open(filter_scores_filepath, "r") as f:
            filter_scores = f.readlines()
            for line in filter_scores[1:]:
                path, neg_prob, pos_prob, prediction = line.strip().split(",")
                if float(pos_prob) >= self.yaml_config.filter_threshold:
                    selected_samples.append(path)

        def get_sample_paths():
            if self.with_masks:
                assert self.config.pseudo_labels_dir is not None
                mask_folder = (
                    Path(self.yaml_config.ukbiobank_masks_dir)
                    / self.config.pseudo_labels_dir
                    / "generated_masks"
                )
                return [
                    (path, mask_folder / path.split("/")[-1])
                    for path in selected_samples
                    if path.endswith(".png")
                ]
            return [(path, None) for path in selected_samples if path.endswith(".png")]

        sample_paths = get_sample_paths()
        return [
            BiobankSampleReference(img_path=img_path, gt_path=gt_path)
            for img_path, gt_path in sample_paths
        ]

    def cv2_loader(self, path: str, is_mask: bool):
        if is_mask:
            img = cv2.imread(path, 0)
            img[img > 0] = 1
        else:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img
