from ai_model.components.model_evaluation import ModelEvaluation
from ai_model.config.configration import ConfigrationManager


class ModelEvaluationPipline:
    STAGE_NAME = "Model Evaluation"

    def __init__(self):
        pass

    def main(self):
        config = ConfigrationManager()

        model_eval_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_eval_config)
        model_evaluation.evaluate()
