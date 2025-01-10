import os

import yaml
from pydantic import BaseModel, Field
import sys

default_value = "<your value here>"


class YamlConfigModel(BaseModel):
    cache_dir: str = Field(
        default="cache",
        description="Directory to store larger temporary files like model checkpoints in",
    )
    wandb_api_key: str = Field(
        description="Your Weights and Biases API key. You can find it in your W&B account settings."
    )
    wandb_project_name: str = Field(description="Your W&B project name.")
    wandb_entity: str = Field(description="Your W&B entity name.")
    results_dir: str = Field(
        default="results", description="Output directory for experiment results"
    )
    ukbiobank_data_dir: str = Field(
        default="/dhc/projects/ukbiobank/derived/imaging/retinal_fundus/images_resized_224",
        description="Directory where UK Biobank data is stored",
    )
    ukbiobank_masks_dir: str = Field(
        default="/dhc/groups/mp2024cl2/retina_masks_ukbiobank",
        description="Directory where UK Biobank masks are stored",
    )
    refuge_dset_path: str = Field(
        default="/dhc/dsets/REFUGE/REFUGE", description="The path to retina dataset"
    )
    drive_dset_path: str = Field(
        default="/dhc/dsets/DRIVE/training", description="The path to the DRIVE dataset"
    )
    chasedb1_dset_path: str = Field(
        default="/dhc/dsets/ChaseDb1",
        description="The path to the CHASEDB1 dataset",
    )
    hrf_dset_path: str = Field(
        default="/dhc/dsets/HRF/", description="The path to the HRF dataset"
    )
    stare_dset_path: str = Field(
        default="/dhc/dsets/STARE-Vessels", description="The path to the STARE dataset"
    )
    filter_dset_path: str = Field(
        default="/dhc/dsets/Retina-Filter",
        description="The path to the FilterDataset",
    )
    filter_threshold: float = Field(
        default=0.2,
        description="Threshold for filtering samples. Samples with a positive probability above this threshold will be considered.",
    )
    aria_dset_path: str = Field(
        default="/dhc/dsets/ARIA",
        description="The path to the ARIA dataset",
    )
    fundus_pixel_mean: tuple[float, float, float] = Field(
        default=(133.54903465404846, 53.65591587936669, 21.037014559695596),
        description="Mean per color channel of fundus images in train partition of JoinedRetinaDataset (RGB)",
    )
    fundus_pixel_std: tuple[float, float, float] = Field(
        default=(81.19483977869938, 35.06718212261534, 14.044927086746483),
        description="Standard deviation per color channel of fundus images in train partition of JoinedRetinaDataset (RGB)",
    )
    fundus_resize_img_size: int = Field(
        default=1024,
        description="Size to which fundus images are resized before training",
    )


class YamlConfig:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            with open(self.config_path, "w") as f:
                print(
                    f"\nCreated a {self.config_path} file in project root. Please replace the autogenerated values in it."
                )
                for name, field in YamlConfigModel.__fields__.items():
                    f.write(
                        f"{name}: {field.default if field.default is not None else default_value}\n"
                    )
                sys.exit(0)
        with open(self.config_path, "r") as f:
            file_content = yaml.safe_load(f)
            try:
                return YamlConfigModel(**file_content)
            except Exception as e:
                raise Exception(
                    f"Error validating fields in config file {self.config_path}: \n{e}"
                )
