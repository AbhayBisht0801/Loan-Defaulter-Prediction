import os
import pandas as pd
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
import joblib 
from src.Loan_defaulter.utils.common import save_json
from urllib.parse import urlparse
import dagshub
from pathlib import Path
from src.Loan_defaulter.config.configuration import MLFlowTrackingConfig


class MLFLOWTracking:
    def __init__(self,config=MLFlowTrackingConfig):
        self.config = config
    def evaL_metrics(self,actual,pred):
        model_precision=precision_score(actual,pred)
        model_recall=recall_score(actual,pred)
        model_accuracy=accuracy_score(actual,pred)
        model_f1_score=accuracy_score(actual,pred)
        return {'accuracy':model_accuracy,'f1_score':model_f1_score,'precision':model_precision,'recall':model_recall}
    def log_into_mlflow(self):
        
        dagshub.init(repo_owner='AbhayBisht0801', repo_name='Loan-Defaulter-Prediction', mlflow=True)
        val_data=pickle.load(open(os.path.join(self.config.test_data_path,'val.pkl'),'rb'))
        X_val=val_data[:,:-1]
        y_val=val_data[:,-1]
        
        model=pickle.load(open(self.config.best_model,'rb'))
        with mlflow.start_run():
            pred=model.predict(X_val)
            metrics=self.evaL_metrics(y_val,pred)
            print(metrics)
            mlflow.log_params(model.get_params())
            save_json(path=Path(self.config.metrics_file_name),data=metrics)
            mlflow.log_metric('accuracy',metrics['accuracy'])
            mlflow.log_metric('f1_score',metrics['f1_score'])
            mlflow.log_metric('precision',metrics['precision'])
            mlflow.log_metric('recall',metrics['recall'])
            
            mlflow.sklearn.log_model(model,"model",registered_model_name="KNN")


        
        

