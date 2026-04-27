from datasets import load_dataset, concatenate_datasets
from pathlib import Path

from ai_model.entity import DataPreparationConfig


class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config

    def main(self):
        config = self.config

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(Path(self.config.train_path)),
                "validation": str(Path(self.config.validation_path)),
                "test_1": str(Path(self.config.test_paths[0])),
                "test_2": str(Path(self.config.test_paths[1])),
            },
        )

        dataset["test"] = concatenate_datasets([dataset["test_1"], dataset["test_2"]])

        dataset.pop("test_1")
        dataset.pop("test_2")

        dataset.remove_columns("section_header")

        dataset.save_to_disk(config.root_dir)
