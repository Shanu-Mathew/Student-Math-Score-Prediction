import os
import sys
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            joblib.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,Y_train,X_test,Y_test,models,param):
    logging.info("Model Evaluation started")
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]
            
            logging.info(f"{model} being executed on dataset")
            gs=GridSearchCV(model,param_grid=para,cv=3)
            gs.fit(X_train,Y_train)  #Train Model

            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)
            Y_train_pred=model.predict(X_train)
            Y_test_pred=model.predict(X_test)

            train_model_score=r2_score(Y_train,Y_train_pred)
            test_model_score=r2_score(Y_test,Y_test_pred)

            report[list(models.keys())[i]]=test_model_score
        
        return report
    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return joblib.load(file_obj)
    
    except:
        raise CustomException(e,sys)