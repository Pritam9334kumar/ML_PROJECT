import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.ml_project.exception import CustomException
from src.ml_project.logger import logging
from src.ml_project.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = { 
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            
        
            params = {
                "Decision Tree": { 
                    "criterion": [
                        "squared_error",
                        "absolute_error", 
                        "poisson"
                    ],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20]

                },

                "Random Forest": {
                    "n_estimators": [8,16,32,64,128,256]
                },


                "Gradient Boosting": {
                    'learning_rate':[0.1, 0.01, 0.05, 0.001],
                    'subsample':[0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8,16,32,64,128,256]

                },

                "Linear Regression": {},

                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 10, 15, 20, 25, 30]
                },

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                'CatBoosting Regressor':{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },

                'AdaBoost Regressor':{
                    'learning_rate':[.1,0.01,0.5,0.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report : dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            #  To get the best model score form the dictionary

            best_model_score = max(model_report.values())

            #  to get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            logging.info(f"Best found model on the both training and testing dataset is { best_model_name}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
                
            )
            predictions = best_model.predict(X_test)
            r2_square = r2_score(y_test, predictions)
            return r2_square

    

        except Exception as e:
            raise CustomException(e, sys)
        
    
        

