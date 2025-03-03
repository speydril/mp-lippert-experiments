from dataclasses import dataclass
from operator import is_
from pathlib import Path

import torch
from src.datasets.ukbiobank_dataset import (
    BiobankBatch,
    BiobankSample,
    UkBiobankDataset,
    UkBiobankDatasetArgs,
)
from src.datasets.joined_retina_dataset import (
    JoinedRetinaDataset,
    JoinedRetinaDatasetArgs,
)
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset, Batch, Sample


@dataclass
class OnlineSTSample(BiobankSample):
    is_gt: bool


@dataclass
class OnlineSTMixedBatch(BiobankBatch):
    is_gt: torch.Tensor


class OnlineSTDatasetArgs(JoinedRetinaDatasetArgs, UkBiobankDatasetArgs):
    gt_step: int = 4


class OnlineSTDataset(BaseDataset):
    def __init__(
        self,
        yaml_config: YamlConfigModel,
        config: OnlineSTDatasetArgs,
        seed: int,
        gt_set: JoinedRetinaDataset,
        uk_biobank_set: UkBiobankDataset,
    ):
        self.config = config
        self.yaml_config = yaml_config
        self.seed = seed
        self.gt_set = gt_set
        self.uk_biobank_set = uk_biobank_set
        self.index_map = self._build_index_map()

    def _build_index_map(self) -> list[tuple[str, int]]:
        biobank_len = len(self.uk_biobank_set)
        gt_len = len(self.gt_set)
        gt_counter = 0
        biobank_counter = 0
        total_counter = 0
        index_map = []
        while biobank_counter < biobank_len:
            if total_counter % self.config.gt_step == 0:
                index_map.append(("gt", gt_counter))
                gt_counter += 1
                # Start back at the beginning
                if gt_counter >= gt_len:
                    gt_counter = 0
            else:
                index_map.append(("biobank", biobank_counter))
                biobank_counter += 1
            total_counter += 1
        return index_map

    def __getitem__(self, index: int) -> OnlineSTSample:
        dataset, sample_index = self.index_map[index]
        if dataset == "gt":
            sample = self.gt_set[sample_index]
            return OnlineSTSample(
                input=sample.input,
                target=sample.target,
                original_size=torch.Tensor(tuple(sample.input.shape[1:3])),
                image_size=torch.Tensor(tuple(sample.input.shape[1:3])),
                img_path=Path("gt"),
                gt_path=None,
                is_gt=False,
            )
        else:
            sample = self.uk_biobank_set[sample_index]
            return OnlineSTSample(
                input=sample.input,
                target=sample.target,
                original_size=sample.original_size,
                image_size=sample.image_size,
                img_path=sample.img_path,
                gt_path=sample.gt_path,
                is_gt=False,
            )

    def get_collate_fn(self):  # type: ignore
        def collate(samples: list[OnlineSTSample]):
            inputs = torch.stack([s.input for s in samples])
            targets = torch.stack([s.target for s in samples])
            original_size = torch.stack([s.original_size for s in samples])
            image_size = torch.stack([s.image_size for s in samples])
            is_gt = torch.Tensor([s.is_gt for s in samples])
            return OnlineSTMixedBatch(
                inputs,
                targets,
                original_size=original_size,
                image_size=image_size,
                file_paths=[s.img_path for s in samples],
                gt_paths=[s.gt_path for s in samples],
                is_gt=is_gt,
            )

        return collate

    def __len__(self) -> int:
        if self.config.limit is not None:
            return min(self.config.limit, len(self.index_map))
        return len(self.index_map)
