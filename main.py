from ai_model.logging import logger
from ai_model.pipline.stage_01_data_ingestion import DataIngestionTrainigPipline
from ai_model.pipline.stage_02_data_validation import DataValidationTrainigPipline
from ai_model.pipline.stage_03_data_transformation import DataTransformationTrainigPipline
from ai_model.pipline.stage_04_model_trainer import ModelTrainerTrainingPipline
from ai_model.pipline.stage_05_model_evaluation import ModelEvaluationTrainingPipline

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    data_ingestion = DataIngestionTrainigPipline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<<\n\n\t\tx=================x")
except Exception as e:
    logger.exception(e)
    raise e


# STAGE_NAME = "Data Validation stage"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
#     data_validation = DataValidationTrainigPipline()
#     data_validation.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<<\n\n\t\tx=================x")
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME = "Data Tranformation stage"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
#     data_tranformation = DataTransformationTrainigPipline()
#     data_tranformation.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<<\n\n\t\tx=================x")
# except Exception as e:
#     logger.exception(e)
#     raise e


logger.info("\t\t\t\tMODEL EVAL START\t\t\t\t\n\n")

# STAGE_NAME = "Model Trainer stage"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
#     model_trainer = ModelTrainerTrainingPipline()
#     model_trainer.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<<\n\n\t\tx=================x")
# except Exception as e:
#     logger.exception(e)
#     raise e



# STAGE_NAME = "Model Evaluation stage"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
#     model_evaluation = ModelEvaluationTrainingPipline()
#     model_evaluation.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<<\n\n\t\tx=================x")
# except Exception as e:
#     logger.exception(e)
#     raise e
# logger.info("\n\n\t\t\t\tMODEL EVAL END\t\t\t\t\n\n")
