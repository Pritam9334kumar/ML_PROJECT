import os
import sys
from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import pymysql
import pandas as pd
from dotenv import load_dotenv

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