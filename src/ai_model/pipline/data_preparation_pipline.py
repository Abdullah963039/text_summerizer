from ai_model.components.data_preparation import DataPreparation
from ai_model.config.configration import ConfigrationManager


class DataPreparationPipline:
    STAGE_NAME = "Data Preparation"

    def __init__(self):
        pass

    def main(self):
        config = ConfigrationManager()
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        data_preparation.main()
