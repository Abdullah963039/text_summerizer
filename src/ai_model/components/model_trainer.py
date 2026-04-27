import os
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import torch

from ai_model.entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # loading data
        dataset = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,  # Main output directory
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=50,  # Save checkpoint every 500 steps (instead of 1e6)
            save_total_limit=3,  # Keep only last 3 checkpoints to save disk space
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            load_best_model_at_end=True,  # Load best model at end of training
            metric_for_best_model="eval_loss",  # Metric to determine best model
            greater_is_better=False,  # Lower loss is better
            save_safetensors=True,  # Save in safetensors format (safer)
            save_only_model=False,  # Save optimizer and scheduler state too
        )

        trainer = Trainer(
            model=model,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
        )

        # Resume training from latest checkpoint if exists
        last_checkpoint = self.get_last_checkpoint(self.config.root_dir)
        if last_checkpoint:
            print(f"Resuming training from checkpoint: {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()

        ## Save final model
        model.save_pretrained(
            os.path.join(self.config.root_dir, self.config.save_model)
        )
        ## Save final tokenizer
        tokenizer.save_pretrained(
            os.path.join(self.config.root_dir, self.config.save_tokenizer)
        )
        
        # Save training arguments and metrics
        self.save_training_metadata(trainer, tokenizer)
    
    def get_last_checkpoint(self, output_dir):
        """Find the latest checkpoint in the output directory"""
        import glob
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if checkpoints:
            # Get the checkpoint with highest step number
            return max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        return None
    
    def save_training_metadata(self, trainer, tokenizer):
        """Save training metadata and metrics"""
        import json
        
        # Save training state
        if hasattr(trainer, 'state'):
            state_path = os.path.join(self.config.root_dir, "training_state.json")
            with open(state_path, 'w') as f:
                json.dump({
                    "global_step": trainer.state.global_step,
                    "epoch": trainer.state.epoch,
                    "best_metric": trainer.state.best_metric,
                    "best_model_checkpoint": trainer.state.best_model_checkpoint,
                    "log_history": trainer.state.log_history
                }, f, indent=2)
        
        # Save model config
        config_path = os.path.join(self.config.root_dir, "model_config.json")
        trainer.model.config.to_json_file(config_path)