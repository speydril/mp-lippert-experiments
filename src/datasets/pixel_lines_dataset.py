from dataclasses import dataclass
from pathlib import Path
import random
from src.models.auto_sam_model import SAMBatch, SAMSampleFileReference
from src.util.polyp_transform import get_polyp_transform
import numpy as np
import cv2
from src.datasets.base_dataset import BaseDataset, Sample
from src.models.segment_anything.utils.transforms import ResizeLongestSide
from pydantic import BaseModel, Field
from src.args.yaml_config import YamlConfigModel
from typing import Literal, Optional
import torch
from typing_extensions import Self
import os
from PIL import Image, ImageDraw

from src.util.datatset_helper import suggest_split


@dataclass
class PixelLinesSample(Sample):
    original_size: torch.Tensor
    image_size: torch.Tensor


@dataclass
class PixelLinesSampleReference:
    split: str
    img: Image.Image


class PixelLinesDatasetArgs(BaseModel):
    """Define arguments for the dataset here, i.e. preprocessing related stuff etc"""

    pixel_lines_test_equals_val: bool = Field(
        default=True,
        description="Whether the test set should be the same as the val set",
    )
    pixel_lines_train_percentage: float = Field(
        default=0.8,
        description="Percentage of data to use for training. Other data will be assigned to val and, if enabled, test.",
    )
    augment_train: bool = True
    line_width: int = 2
    sample_count: int = 20
    resolution: int = 500
    line_color: str = "blue"


class PixelLinesDataset(BaseDataset):
    def __init__(
        self,
        config: PixelLinesDatasetArgs,
        yaml_config: YamlConfigModel,
        samples: Optional[list[PixelLinesSampleReference]] = None,
    ):
        self.yaml_config = yaml_config
        self.config = config
        self.sam_trans = ResizeLongestSide(
            self.yaml_config.fundus_resize_img_size,
        )
        self.samples = (
            [
                PixelLinesSampleReference(
                    split=suggest_split(
                        i,
                        self.config.sample_count,
                        self.config.pixel_lines_train_percentage,
                    ),
                    img=self.generate_sample(),
                )
                for i in range(self.config.sample_count)
            ]
            if samples is None
            else samples
        )

    def __getitem__(self, index: int) -> PixelLinesSample:
        img = self.samples[index].img
        gt = self.get_gt_for_image(img)
        train_transform, test_transform = get_polyp_transform()

        augmentations = test_transform
        img, mask = augmentations(np.array(img), gt.transpose(0,1))


        original_size = tuple(img.shape[1:3])
        img, mask = self.sam_trans.apply_image_torch(
            torch.Tensor(img)
        ), self.sam_trans.apply_image_torch(torch.Tensor(mask))
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        image_size = tuple(img.shape[1:3])

        return PixelLinesSample(
            input=self.sam_trans.preprocess(img),
            target=self.sam_trans.preprocess(mask),
            original_size=torch.Tensor(original_size),
            image_size=torch.Tensor(image_size),
        )

    def __len__(self):
        return len(self.samples)

    def get_collate_fn(self):  # type: ignore
        def collate(samples: list[PixelLinesSample]):
            inputs = torch.stack([s.input for s in samples])
            targets = torch.stack([s.target for s in samples])
            original_size = torch.stack([s.original_size for s in samples])
            image_size = torch.stack([s.image_size for s in samples])
            return SAMBatch(
                inputs, targets, original_size=original_size, image_size=image_size
            )

        return collate

    def get_split(self, split: Literal["train", "val", "test"]) -> Self:
        if self.config.pixel_lines_test_equals_val and split == "test":
            split = "val"
        return self.__class__(
            self.config,
            self.yaml_config,
            [s for s in self.samples if s.split == split],
        )

    def generate_sample(self):
        img = Image.new(
            "RGB", (self.config.resolution, self.config.resolution), "white"
        )
        draw = ImageDraw.Draw(img)

        points = [
            (
                random.randint(0, self.config.resolution),
                random.randint(0, self.config.resolution),
            )
            for _ in range(5)
        ]
        draw.line(points, fill=self.config.line_color, width=self.config.line_width)

        return img

    def get_gt_for_image(self, img: Image.Image):
        img_array = np.array(img)
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        non_white_pixels = (
            (img_array[:, :, 0] != 255)
            | (img_array[:, :, 1] != 255)
            | (img_array[:, :, 2] != 255)
        )
        mask[non_white_pixels] = 1
        return mask
