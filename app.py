from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
from src.ml_project.components.data_ingestion import DataIngestion
from src.ml_project.components.data_transformation import DataTransformation
from src.ml_project.components.model_trainer import ModelTrainer

import sys

if __name__ == "__main__":
    logging.info("Starting the application")

    try:

        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()

        train_array, test_array, _ = (
            data_transformation.initiate_data_transformation(
                train_data_path,
                test_data_path
            )
        )

        # Model Training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_array, test_array))

    except Exception as e:
        logging.info("Exception occurred")
        raise CustomException(e, sys)