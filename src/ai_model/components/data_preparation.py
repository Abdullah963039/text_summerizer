from datasets import load_dataset, concatenate_datasets
from pathlib import Path

from ai_model.entity import DataPreparationConfig


class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config

    def main(self):
        config = self.config

        train_ds_path = Path(f"{config.data_path}/{config.train_filename}")
        validation_ds_path = Path(f"{config.data_path}/{config.validation_filename}")
        test1_ds_path = Path(f"{config.data_path}/{config.test_filenames[0]}")
        test2_ds_path = Path(f"{config.data_path}/{config.test_filenames[1]}")

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(train_ds_path),
                "validation": str(validation_ds_path),
                "test_1": str(test1_ds_path),
                "test_2": str(test2_ds_path),
            },
        )

        dataset["test"] = concatenate_datasets([dataset["test_1"], dataset["test_2"]])

        dataset.pop("test_1")
        dataset.pop("test_2")
        
        dataset.save_to_disk(config.root_dir)