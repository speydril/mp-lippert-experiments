from ast import Not
from dataclasses import dataclass
from pathlib import Path
import re
from src.models.auto_sam_model import SAMBatch, SAMSampleFileReference
from src.util.polyp_transform import get_polyp_transform
import src.util.transforms_shir as transforms
import numpy as np
import cv2
from src.datasets.base_dataset import BaseDataset, Batch, Sample
from src.models.segment_anything.utils.transforms import ResizeLongestSide
from torchvision.datasets import MNIST
from pydantic import BaseModel
from src.args.yaml_config import YamlConfigModel
from typing import Callable, Literal, Optional
from math import floor
import torch
from typing_extensions import Self
import os
from PIL import Image

from src.util.datatset_helper import suggest_split


@dataclass
class ARIASample(Sample):
    original_size: torch.Tensor
    image_size: torch.Tensor
    idx: int


class ARIADatasetArgs(BaseModel):
    """Define arguments for the dataset here, i.e. preprocessing related stuff etc"""

    aria_annotator: Literal["BSS", "BDP"] = "BSS"


@dataclass
class ARIASampleFileReference(SAMSampleFileReference):
    split: str


class ARIADataset(BaseDataset):
    def __init__(
        self,
        config: ARIADatasetArgs,
        yaml_config: YamlConfigModel,
        samples: Optional[list[ARIASampleFileReference]] = None,
    ):
        self.yaml_config = yaml_config
        self.config = config
        self.samples = self.load_data() if samples is None else samples
        pixel_mean, pixel_std = (
            self.yaml_config.fundus_pixel_mean,
            self.yaml_config.fundus_pixel_std,
        )
        self.sam_trans = ResizeLongestSide(
            self.yaml_config.fundus_resize_img_size, pixel_mean, pixel_std
        )

    def __getitem__(self, index: int) -> ARIASample:
        sample = self.samples[index]
        train_transform, test_transform = get_polyp_transform()

        augmentations = test_transform

        image = self.cv2_loader(sample.img_path, is_mask=False)
        gt = self.cv2_loader(sample.gt_path, is_mask=True)

        img, mask = augmentations(image, gt)

        original_size = tuple(img.shape[1:3])
        img, mask = self.sam_trans.apply_image_torch(
            torch.Tensor(img)
        ), self.sam_trans.apply_image_torch(torch.Tensor(mask))
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        image_size = tuple(img.shape[1:3])

        return ARIASample(
            input=self.sam_trans.preprocess(img),
            target=self.sam_trans.preprocess(mask),
            original_size=torch.Tensor(original_size),
            image_size=torch.Tensor(image_size),
            idx=index,
        )

    def __len__(self):
        return len(self.samples)

    def get_collate_fn(self):  # type: ignore
        def collate(samples: list[ARIASample]):
            inputs = torch.stack([s.input for s in samples])
            targets = torch.stack([s.target for s in samples])
            original_size = torch.stack([s.original_size for s in samples])
            image_size = torch.stack([s.image_size for s in samples])
            return SAMBatch(
                inputs, targets, original_size=original_size, image_size=image_size
            )

        return collate

    def get_split(self, split: Literal["train", "val", "test"]) -> Self:
        return self.__class__(
            self.config,
            self.yaml_config,
            [s for s in self.samples if s.split == split],
        )

    def load_data(self):
        imgs_dir = os.path.join(self.yaml_config.aria_dset_path, "images")
        gts_dir = os.path.join(self.yaml_config.aria_dset_path, "masks")

        images_and_masks_paths = [
            (
                str(Path(imgs_dir) / img_file),
                str(
                    Path(gts_dir)
                    / f"{img_file.removesuffix('.tif')}_{self.config.aria_annotator}.tif"
                ),
            )
            for img_file in os.listdir(imgs_dir)
        ]

        return [
            ARIASampleFileReference(
                img_path=img,
                gt_path=mask,
                split=suggest_split(i, len(images_and_masks_paths), 0.8, False),
            )
            for i, (img, mask) in enumerate(images_and_masks_paths)
        ]

    def cv2_loader(self, path, is_mask):
        if is_mask:
            with Image.open(path) as im:
                img = np.array(im.convert("L"))
            img[img > 0] = 1
        else:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img
