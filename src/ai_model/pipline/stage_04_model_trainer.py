from ai_model.components.model_trainer import ModelTrainer
from ai_model.config.configration import ConfigrationManager


class ModelTrainerTrainingPipline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigrationManager()

        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()
