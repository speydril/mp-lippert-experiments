from dataclasses import dataclass
from typing import Callable, Literal, Optional, cast
import torch
from typing_extensions import Self
from src.datasets.joined_retina_dataset import JoinedRetinaDatasetArgs
from src.datasets.aria_dataset import ARIADataset
from src.models.auto_sam_model import SAMSampleFileReference
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset, JoinedDataset, Sample
from src.datasets.chasedb1_dataset import (
    ChaseDb1Dataset,
)
from src.datasets.drive_dataset import DriveDataset
from src.datasets.hrf_dataset import HrfDataset
from src.datasets.stare_dataset import STAREDataset
from math import floor

from src.util.polyp_transform import get_polyp_transform
from src.util.image_util import extract_patch


@dataclass
class PatchedVesselSample(Sample):
    original_size: torch.Tensor
    image_size: torch.Tensor
    origin_dataset: str


class PatchedVesselDataset(BaseDataset):
    def __init__(
        self,
        ds: DriveDataset | HrfDataset | STAREDataset | ChaseDb1Dataset | ARIADataset,
        augment_train: bool = True,
        patches: Literal[4, 16] = 4,
    ):
        self.ds = ds
        self.samples = ds.samples
        self.augment_train = augment_train
        self.patches = patches

    def __len__(self) -> int:
        return len(self.ds) * (self.patches)

    def __getitem__(self, index: int) -> Sample:
        sample = self.ds.samples[floor(index / self.patches)]
        train_transform, test_transform = get_polyp_transform()

        augmentations = (
            train_transform
            if sample.split == "train" and self.augment_train
            else test_transform
        )

        image = self.ds.cv2_loader(sample.img_path, is_mask=False)
        gt = self.ds.cv2_loader(sample.gt_path, is_mask=True)
        # image shape: (H,W,3), gt shape: (H,W)
        quadrant_id = index % 4
        image, gt = extract_patch(image, quadrant_id), extract_patch(gt, quadrant_id)
        if self.patches == 16:
            subquadrant_id = (index // 4) % 4
            image, gt = extract_patch(image, subquadrant_id), extract_patch(
                gt, subquadrant_id
            )

        img, mask = augmentations(image, gt)

        original_size = tuple(img.shape[1:3])
        img, mask = self.ds.sam_trans.apply_image_torch(
            torch.Tensor(img)
        ), self.ds.sam_trans.apply_image_torch(torch.Tensor(mask))
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        image_size = tuple(img.shape[1:3])

        return PatchedVesselSample(
            input=self.ds.sam_trans.preprocess(img),
            target=self.ds.sam_trans.preprocess(mask),
            original_size=torch.Tensor(original_size),
            image_size=torch.Tensor(image_size),
            origin_dataset=self.ds.__class__.__name__,
        )

    def get_split(self, split: Literal["train", "val", "test"]) -> Self:
        return self.__class__(self.ds.get_split(split))


class JoinedPatchedRetinaDataset(JoinedDataset):
    def __init__(
        self,
        datasets: list[PatchedVesselDataset],
        collate: Optional[Callable] = None,
        limit_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.limit_samples = limit_samples
        super().__init__(datasets, collate, seed)  # type: ignore
        self.datasets = cast(list[PatchedVesselDataset], self.datasets)

    def get_file_refs(self) -> list[SAMSampleFileReference]:
        return [sample for ds in self.datasets for sample in ds.samples]

    @classmethod
    def from_config(
        cls,
        config: JoinedRetinaDatasetArgs,
        yaml_config: YamlConfigModel,
        seed: int,
        patches: Literal[4, 16],
    ):
        drive = DriveDataset(config=config, yaml_config=yaml_config)
        chase_db1 = ChaseDb1Dataset(config=config, yaml_config=yaml_config)
        hrf = HrfDataset(config=config, yaml_config=yaml_config)
        stare = STAREDataset(config=config, yaml_config=yaml_config)

        datasets = [drive, chase_db1, hrf, stare]
        if config.include_aria:
            aria = ARIADataset(config=config, yaml_config=yaml_config)
            datasets.append(aria)
        return cls(
            [
                PatchedVesselDataset(
                    ds, augment_train=config.augment_train, patches=patches
                )
                for ds in datasets
            ],
            drive.get_collate_fn(),
            seed=seed,
        )

    def __len__(self) -> int:
        if self.limit_samples is not None:
            return self.limit_samples
        return super().__len__()

    def get_split(
        self,
        split: Literal["train", "val", "test"],
        limit_samples: Optional[int] = None,
    ) -> Self:
        return self.__class__(
            [dataset.get_split(split) for dataset in self.datasets],
            self.collate,
            limit_samples=limit_samples,
            seed=self.seed,
        )
