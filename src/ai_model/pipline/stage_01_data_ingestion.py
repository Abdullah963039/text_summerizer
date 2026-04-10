from ai_model.components.data_ingestion import DataIngestion
from ai_model.config.configration import ConfigrationManager


class DataIngestionTrainigPipline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigrationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
