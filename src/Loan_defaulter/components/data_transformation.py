import os
import urllib.request as request 
import gdown
from src.Loan_defaulter import logger
import zipfile
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from src.Loan_defaulter.utils.common import save_object
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.Loan_defaulter.entity.config_entity import DataTransformationConfig
class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config = config
        
       
    def Transformation(self):
        try:
            logger.info('Creating Pipeline')
            X=pd.read_csv(os.path.join(self.config.data_dir,'X.csv'),index_col=False)
            

            numerical=[]
            categorical=[]
            for i in X.columns:
                if X[i].dtype=='object':
                    categorical.append(i)
                else:
                    if len(X[i].unique())<30:
                        categorical.append(i)
                    else:
                    
                        numerical.append(i)
            age=['age']
            categorical.remove('age')
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
            ])

            Age = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ('Ordinal_encode', OrdinalEncoder(categories=[['<25', '25-34', '35-44', '45-54', '55-64', '65-74', '>74']]))
            ])
            preprocessing=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical),
                    ('cat_pipeline',cat_pipeline,categorical),
                    ('age_oe',Age,age)
                ]
            )
            logger.info('Pipeline Created')
            return preprocessing
        except Exception as e:
            raise e  
    def split_data_transform(self):
        try:
            print(self.config)
            logger.info('Executing the pipeline on data')
            X=pd.read_csv(os.path.join(self.config.data_dir,'X.csv'),index_col=False)

            y=pd.read_csv(os.path.join(self.config.data_dir,'y.csv'),index_col=False)
            X_train,X_temp,y_train,y_temp=train_test_split(X,y,random_state=42,test_size=0.3)
            X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,random_state=42,test_size=0.5)
            preprocessing_obj=self.Transformation()
            X_train_preprocessed=preprocessing_obj.fit_transform(X_train)
            X_test_preprocessed=preprocessing_obj.transform(X_test)
            X_val_preprocessed=preprocessing_obj.transform(X_val)
            save_object(file_path=os.path.join(self.config.preprocess_obj,'pipeline.pkl'),
                        obj=preprocessing_obj
            )
            train=np.c_[X_train_preprocessed,y_train.to_numpy()]
            test=np.c_[X_test_preprocessed,y_test.to_numpy()]
            val=np.c_[X_val_preprocessed,y_val.to_numpy()]
            save_object(file_path=os.path.join(self.config.split_dir,'train.pkl'),obj=train)
            save_object(file_path=os.path.join(self.config.split_dir,'val.pkl'),obj=val)
            save_object(file_path=os.path.join(self.config.split_dir,'test.pkl'),obj=test)
            logger.info(f'Executed the pipeline on data and data stored in {self.config.split_dir}')
        except Exception as e:
            raise e