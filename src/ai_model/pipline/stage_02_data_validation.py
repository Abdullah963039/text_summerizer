from ai_model.components.data_validation import DataValidation
from ai_model.config.configration import ConfigrationManager


class DataValidationTrainigPipline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigrationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_files_exists()
