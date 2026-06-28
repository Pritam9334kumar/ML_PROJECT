import sys

from src.ml_project.components.data_ingestion import DataIngestion
from src.ml_project.components.data_transformation import DataTransformation
from src.ml_project.components.model_trainer import ModelTrainer
from src.ml_project.exception import CustomException
from src.ml_project.logger import logging


if __name__ == "__main__":
	logging.info("Starting the training pipeline")

	try:
		data_ingestion = DataIngestion()
		train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

		data_transformation = DataTransformation()
		train_array, test_array, _ = data_transformation.initiate_data_transformation(
			train_data_path,
			test_data_path,
		)

		model_trainer = ModelTrainer()
		r2_score_value = model_trainer.initiate_model_trainer(train_array, test_array)

		print(f"Training completed. Test R2 score: {r2_score_value}")

	except Exception as e:
		logging.info("Exception occurred during training")
		raise CustomException(e, sys)
