import os
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import pymysql
import pandas as pd
from dotenv import load_dotenv
import pickle
import numpy as np



load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading data from mysql database")

    print("HOST =", host)
    print("USER =", user)
    print("DB =", db)
    print("PASSWORD =", password)

    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )

        logging.info("Successfully connected to the database")

        df = pd.read_sql("SELECT * FROM students", mydb)

        print(df.head())

        return df

    except Exception as ex:
        raise CustomException(ex, sys)
    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(
                model,
                param,
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    

    except Exception as e:
        raise CustomException(e, sys)
    

    