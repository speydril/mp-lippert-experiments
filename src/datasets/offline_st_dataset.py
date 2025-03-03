from dataclasses import dataclass
from math import ceil
from src.models.auto_sam_model import SAMBatch
from src.datasets.joined_retina_dataset import (
    JoinedRetinaDataset,
    JoinedRetinaDatasetArgs,
    JoinedRetinaSample,
)
from src.datasets.ukbiobank_dataset import UkBiobankDataset, UkBiobankDatasetArgs
from src.datasets.base_dataset import BaseDataset, Sample
from pydantic import BaseModel, Field
from src.args.yaml_config import YamlConfigModel
from typing import Optional, cast
import torch


@dataclass
class OfflineSTSample(Sample):
    original_size: torch.Tensor
    image_size: torch.Tensor
    origin_ds: str
    origin_ds_index: int


class OfflineStDatasetArgs(BaseModel):
    pseudo_labels_dir: str = Field(
        description="Name of the the directory containing the pseudo labels with pattern <yaml_config.ukbiobank_masks_dir>/<pseudo_labels_dir>/generated_masks/[masks].png"
    )
    filter_scores_filepath: str = (
        "/dhc/groups/mp2024cl2/ukbiobank_filters/filter_predictions.csv"
    )
    labelled_ratio: float = 0.25
    gt_limit: Optional[int] = None


class OfflineSTTrainDataset(BaseDataset):
    def __init__(
        self,
        config: OfflineStDatasetArgs,
        yaml_config: YamlConfigModel,
        seed: int,
    ):
        super().__init__()
        self.yaml_config = yaml_config
        self.config = config

        self.pseudo_train = UkBiobankDataset(
            config=UkBiobankDatasetArgs(
                pseudo_labels_dir=self.config.pseudo_labels_dir,
                train_percentage=1.0,
                val_percentage=0.0,
                test_percentage=0.0,
                filter_scores_filepath=self.config.filter_scores_filepath,
                threshold_pseudo_labels=False,
            ),
            yaml_config=yaml_config,
            with_masks=True,
            augment_inputs=False,
        )
        self.gt_train = JoinedRetinaDataset.from_config(
            JoinedRetinaDatasetArgs(drive_test_equals_val=False), yaml_config, seed
        ).get_split("train", limit_samples=self.config.gt_limit)
        target_n_gt = int(len(self.pseudo_train) * self.config.labelled_ratio)

        gt_index_map = list(range(len(self.gt_train))) * (
            ceil(target_n_gt / len(self.gt_train))
        )
        pseudo_index_map = list(range(len(self.pseudo_train)))

        self.index_map = [("pseudo", i) for i in pseudo_index_map] + [
            ("gt", i) for i in gt_index_map
        ]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int):  # -> OfflineSTSample:
        if index >= len(self):
            raise IndexError
        dataset, sample_index = self.index_map[index]
        if dataset == "pseudo":
            sample = self.pseudo_train[sample_index]
        elif dataset == "gt":
            sample = cast(JoinedRetinaSample, self.gt_train[sample_index])
        else:
            raise ValueError(f"Unknown dataset {dataset}")

        return OfflineSTSample(
            input=sample.input,
            target=sample.target,
            original_size=sample.original_size,
            image_size=sample.image_size,
            origin_ds=dataset,
            origin_ds_index=sample_index,
        )

    def get_collate_fn(self):  # type: ignore
        file_refs = {
            "pseudo": self.pseudo_train.get_file_refs(),
            "gt": self.gt_train.get_file_refs(),
        }

        def collate(samples: list[OfflineSTSample]):
            inputs = torch.stack([s.input for s in samples])
            targets = torch.stack([s.target for s in samples])
            original_size = torch.stack([s.original_size for s in samples])
            image_size = torch.stack([s.image_size for s in samples])
            is_gt = torch.tensor(
                [s.origin_ds == "gt" for s in samples], dtype=torch.bool
            )
            meta = {
                "sample_metadata": [
                    {
                        "origin_dataset": s.origin_ds,
                        "origin_sample_idx": f"{s.origin_ds}_{s.origin_ds_index}",
                        "img_path": file_refs[s.origin_ds][s.origin_ds_index].gt_path,
                    }
                    for s in samples
                ]
            }
            return SAMBatch(
                inputs,
                targets,
                original_size=original_size,
                image_size=image_size,
                metadata=meta,
                is_gt=is_gt,
            )

        return collate

    def get_file_refs(self):
        gt_files = self.gt_train.get_file_refs()
        return [
            self.pseudo_train.samples[i] if dataset == "pseudo" else gt_files[i]
            for (dataset, i) in self.index_map
        ]
