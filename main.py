from ai_model.logging import logger

from ai_model.pipline.data_ingestion_pipline import DataIngestionPipline
from ai_model.pipline.data_preparation_pipline import DataPreparationPipline
from ai_model.pipline.data_validation_pipline import DataValidationPipline
from ai_model.pipline.data_transformation_pipline import (
    DataTransformationPipline,
)
from ai_model.pipline.model_trainer_pipline import ModelTrainerPipline
from ai_model.pipline.model_evaluation_pipline import ModelEvaluationPipline


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    data_ingestion = DataIngestionPipline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<<\n\n\t\tx=================x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Preparation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    data_preparation = DataPreparationPipline()
    data_preparation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<<\n\n\t\tx=================x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    data_validation = DataValidationPipline()
    data_validation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<<\n\n\t\tx=================x")
except Exception as e:
    logger.exception(e)
    raise e


# STAGE_NAME = "Data Tranformation stage"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
#     data_tranformation = DataTransformationPipline()
#     data_tranformation.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<<\n\n\t\tx=================x")
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME = "Model Trainer stage"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
#     model_trainer = ModelTrainerPipline()
#     model_trainer.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<<\n\n\t\tx=================x")
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME = "Model Evaluation stage"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
#     model_evaluation = ModelEvaluationPipline()
#     model_evaluation.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<<\n\n\t\tx=================x")
# except Exception as e:
#     logger.exception(e)
#     raise e
# logger.info("\n\n\t\t\t\tMODEL EVAL END\t\t\t\t\n\n")
