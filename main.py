from src.Loan_defaulter import logger
from src.Loan_defaulter.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.Loan_defaulter.pipeline.stage_02_data_transformation import DataTransformationPipeline
STAGE_NAME="Data Ingestion stage"
try:
    logger.info(f'>>>>>>stage {STAGE_NAME} started <<<<<<<')
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>stage {STAGE_NAME} completed <<<<<<<')
except Exception as e:
    logger.exception(e)
    raise e
STAGE_NAME="Data Transformation stage"
try:
    logger.info(f'>>>>>>stage {STAGE_NAME} started <<<<<<<')
    obj=DataTransformationPipeline()
    obj.main()
    logger.info(f'>>>>>>stage {STAGE_NAME} completed <<<<<<<')
except Exception as e:
    logger.exception(e)
    raise e