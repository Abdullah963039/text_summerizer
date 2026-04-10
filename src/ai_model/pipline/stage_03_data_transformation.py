from ai_model.components.data_tranformation import DataTransformation
from ai_model.config.configration import ConfigrationManager


class DataTransformationTrainigPipline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigrationManager()

        data_tranfromation_config = config.get_data_transformation_config()
        data_tranfromation = DataTransformation(config=data_tranfromation_config)
        data_tranfromation.convert()
