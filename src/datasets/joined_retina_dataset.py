from typing import Callable, Literal, Optional
from typing_extensions import Self
from pydantic import BaseModel, Field
from src.datasets.aria_dataset import ARIADataset, ARIADatasetArgs
from src.models.auto_sam_model import SAMSampleFileReference
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset, JoinedDataset
from src.datasets.chasedb1_dataset import ChaseDb1Dataset, ChaseDb1DatasetArgs
from src.datasets.drive_dataset import DriveDataset, DriveDatasetArgs
from src.datasets.hrf_dataset import HrfDataset, HrfDatasetArgs
from src.datasets.stare_dataset import STAREDataset, STAREDatasetArgs


class JoinedRetinaDatasetArgs(
    DriveDatasetArgs,
    ChaseDb1DatasetArgs,
    STAREDatasetArgs,
    HrfDatasetArgs,
    ARIADatasetArgs,
):
    include_aria: bool = Field(
        default=True, description="Include ARIA dataset in the joined dataset"
    )


class JoinedRetinaDataset(JoinedDataset):
    def __init__(
        self,
        datasets: list[BaseDataset],
        collate: Optional[Callable] = None,
        limit_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.limit_samples = limit_samples
        super().__init__(datasets, collate, seed)  # type: ignore

    def get_file_refs(self) -> list[SAMSampleFileReference]:
        return [sample for ds in self.datasets for sample in ds.samples]  # type: ignore

    @classmethod
    def from_config(
        cls, config: JoinedRetinaDatasetArgs, yaml_config: YamlConfigModel, seed: int
    ):
        drive = DriveDataset(config=config, yaml_config=yaml_config)
        chase_db1 = ChaseDb1Dataset(config=config, yaml_config=yaml_config)
        hrf = HrfDataset(config=config, yaml_config=yaml_config)
        stare = STAREDataset(config=config, yaml_config=yaml_config)

        datasets = [drive, chase_db1, hrf, stare]
        if config.include_aria:
            aria = ARIADataset(config=config, yaml_config=yaml_config)
            datasets.append(aria)
        return cls(datasets, drive.get_collate_fn(), seed=seed)

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
