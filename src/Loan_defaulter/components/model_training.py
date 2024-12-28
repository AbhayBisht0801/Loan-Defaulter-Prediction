from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.Loan_defaulter.utils.common import load_object,evaluate_models,save_object
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from xgboost import XGBClassifier
from src.Loan_defaulter.config.configuration import ModelTrainingConfig
class ModelTrainer:
    def __init__(self,config:ModelTrainingConfig):
        self.config = config
    def load_datasets(self):
        self.training_data=load_object(os.path.join(self.config.split_dir,'train.pkl'))
        self.testing_data=load_object(os.path.join(self.config.split_dir,'test.pkl'))
        self.val_data=load_object(os.path.join(self.config.split_dir,'val.pkl'))
    def training(self):
        X_train,y_train,X_test,y_test,X_val,y_val=(
            self.training_data[:,:-1],self.training_data[:,-1],
            self.testing_data[:,:-1],self.testing_data[:,-1],
            self.val_data[:,:-1],self.val_data[:,-1]

        )
        

        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        negative_samples = sum(y_train == 0)
        positive_samples = sum(y_train == 1)
        scale_pos_weight = negative_samples / positive_samples

# Use scale_pos_weight in the model

        models={
            'BalancedKNN':BalancedBaggingClassifier(estimator=KNeighborsClassifier(n_jobs=-1),random_state=42),
            'BalancedLogistic':BalancedBaggingClassifier(estimator=LogisticRegression(n_jobs=-1)),
            'BalancedRandomForest':BalancedRandomForestClassifier(n_jobs=-1),
            'LogisticRegression':LogisticRegression(class_weight=class_weight_dict,n_jobs=-1),
            'XGC':XGBClassifier(scale_pos_weight=scale_pos_weight),
            'DT':DecisionTreeClassifier(class_weight=class_weight_dict),
            'RandomForestClassifier':RandomForestClassifier(class_weight=class_weight_dict)
        }
# Evaluate models
        model_report = evaluate_models(X=X_train, y=y_train, models=models, X_test=X_test, y_test=y_test)

        # Sort the models based on the highest metric value
        sorted_models = dict(sorted(model_report.items(), key=lambda item: max(item[1]), reverse=True))

        # Filter out overfitting models (assuming overfitting means a perfect score of 1 on training)
        non_overfit_models = {
            model: metrics for model, metrics in model_report.items() if metrics[0] != 1 and metrics[1] != 1
        }

        # Get the best non-overfitting model (sorted by the maximum metric value)
        best_model_name, best_model_metrics = sorted(non_overfit_models.items(), key=lambda item: max(item[1]), reverse=True)[0]

        # Output the best model's name and metrics
        print(f"Best Model: {best_model_name}")
        print(f"Metrics: {best_model_metrics}")
        best_model=models[best_model_name]
        save_object(self.config.best_model,obj=best_model)
        print('recall',best_model_metrics[0],'precision',best_model_metrics[1])


        