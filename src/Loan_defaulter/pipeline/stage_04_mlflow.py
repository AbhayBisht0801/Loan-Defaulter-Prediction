
from src.Loan_defaulter.config.configuration import ConfigurationManager
from src.Loan_defaulter.components.mlflow_tracking import MLFLOWTracking
from src.Loan_defaulter import logger


STAGE_NAME="Mlflow tracking stage"

class MLFLOWTrackingPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        mlflow_config=config.get_data_training_config()
        mlflow_tracking=MLFLOWTracking(config=mlflow_config)
        mlflow_tracking.log_into_mlflow()
if __name__=='__main__':
    try:
        logger.info(f'>>>>>>stage {STAGE_NAME} started <<<<<<<')
        obj=MLFLOWTrackingPipeline()
        obj.main()
        logger.info(f'>>>>>>stage {STAGE_NAME} completed <<<<<<<')
    except Exception as e:
        logger.exception(e)
        raise e