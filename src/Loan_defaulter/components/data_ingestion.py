import os
import urllib.request as request 
import gdown
from src.Loan_defaulter import logger
import zipfile
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from src.Loan_defaulter.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config

    def download_file(self)->str:
        try:
            dataset_url=self.config.source_URL
            local_dir=self.config.local_dir
            os.makedirs('artifacts/data_ingestion',exist_ok=True)
            logger.info(f'Downloading data from {dataset_url} into file {local_dir}')
            file_id=dataset_url.split('/')[-2]
            prefix='http://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,local_dir)
            logger.info(f'Download data from {dataset_url} into file {local_dir}')
        except Exception as e:
            raise e
    def split_data(self):
        try:
            logger.info('Splitting data into train test split')
            df=pd.read_csv(self.config.local_dir)
            df.drop(columns=['ID','year'],inplace=True)
            X=df.drop(columns='Status')
            y=df['Status']
            X.to_csv(os.path.join(self.config.data_dir,'X.csv'),index=False)
            y.to_csv(os.path.join(self.config.data_dir,'y.csv'),index=False)
            
            
        except Exception as e:
            raise e

        
        
    
    