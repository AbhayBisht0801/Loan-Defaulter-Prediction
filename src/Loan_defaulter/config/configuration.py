from src.Loan_defaulter.constants import *
from src.Loan_defaulter.utils.common import read_yaml,create_directories
from src.Loan_defaulter.entity.config_entity import DataIngestionConfig,DataTransformationConfig 
class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH

    ):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)->DataIngestionConfig:
        config=self.config.data_ingestion
       
        create_directories([config.root_dir])
        create_directories([config.data_dir])
        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_dir=config.local_dir,
            data_dir=config.data_dir
        )
        return data_ingestion_config

    def get_data_transformation_config(self)->DataTransformationConfig:
        print(self.config)
        config=self.config
       
        create_directories([config.data_transformation.root_dir])
        create_directories([config.data_transformation.split_dir])
        create_directories([config.data_transformation.preprocess_obj])
        data_transformation_config=DataTransformationConfig(
            root_dir=config.data_transformation.root_dir,
            split_dir=config.data_transformation.split_dir,
            preprocess_obj=config.data_transformation.preprocess_obj,
            data_dir=config.data_ingestion.data_dir
        )
        return data_transformation_config