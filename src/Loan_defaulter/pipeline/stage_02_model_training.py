from src.Loan_defaulter.config.configuration import ConfigurationManager
from src.Loan_defaulter.components.model_training import ModelTrainer
from src.Loan_defaulter import logger


STAGE_NAME="Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        training_config=config.get_data_training_config()
        train=ModelTrainer(config=training_config)
        train.load_datasets()
        train.training()
if __name__=='__main__':
    try:
        logger.info(f'>>>>>>stage {STAGE_NAME} started <<<<<<<')
        obj=ModelTrainer
        obj.main()
        logger.info(f'>>>>>>stage {STAGE_NAME} completed <<<<<<<')
    except Exception as e:
        logger.exception(e)
        raise e