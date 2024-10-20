import os

import yaml
from pydantic import BaseModel, Field

default_value = "<your value here>"


class YamlConfigModel(BaseModel):
    cache_dir: str = Field(
        description="Directory to store larger temporary files like model checkpoints in"
    )
    wandb_api_key: str = Field(
        description="Your Weights and Biases API key. You can find it in your W&B account settings."
    )
    wandb_project_name: str = Field(description="Your W&B project name.")
    wandb_entity: str = Field(description="Your W&B entity name.")
    results_dir: str = Field(description="Output directory for experiment results")


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
                exit(0)
        with open(self.config_path, "r") as f:
            file_content = yaml.safe_load(f)
            try:
                return YamlConfigModel(**file_content)
            except Exception as e:
                raise Exception(
                    f"Error validating fields in config file {self.config_path}: \n{e}"
                )
