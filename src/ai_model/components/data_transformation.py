import os
from transformers import AutoTokenizer
from datasets import load_from_disk
from ai_model.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        # Set pad_token to eos_token for Pegasus
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def convert_examples_to_features(self, example_batch):
        c = self.config

        input_encondings = self.tokenizer(
            example_batch["dialogue"],
            max_length=c.tokenizer_input_max_length,
            truncation=c.tokenizer_truncation,
            padding=c.tokenizer_padding,
        )

        target_encondings = self.tokenizer(
            example_batch["section_text"],
            max_length=c.tokenizer_target_max_length,
            truncation=c.tokenizer_truncation,
            padding=c.tokenizer_padding,
        )

        return {
            "input_ids": input_encondings["input_ids"],
            "attention_mask": input_encondings["attention_mask"],
            "labels": target_encondings["input_ids"],
        }

    def convert(self):
        config = self.config

        dataset_samsum = load_from_disk(config.data_path)
        dataset_samsum_pt = dataset_samsum.map(
            self.convert_examples_to_features,
            batched=True,
            remove_columns="section_header",
        )
        dataset_samsum_pt.save_to_disk(
            os.path.join(config.root_dir, config.save_dataset_dir)
        )
