import sys
import os
from dataclasses import dataclass

import pandas as pd

from src.ml_project.exception import CustomException
from src.ml_project.utils import load_object


@dataclass
class CustomData:
	gender: str
	race_ethnicity: str
	parental_level_of_education: str
	lunch: str
	test_preparation_course: str
	reading_score: float
	writing_score: float

	def get_data_as_data_frame(self):
		data_dict = {
			"gender": [self.gender],
			"race_ethnicity": [self.race_ethnicity],
			"parental_level_of_education": [self.parental_level_of_education],
			"lunch": [self.lunch],
			"test_preparation_course": [self.test_preparation_course],
			"reading_score": [self.reading_score],
			"writing_score": [self.writing_score],
		}

		return pd.DataFrame(data_dict)


class PredictPipeline:
	def __init__(self):
		self.model_path = os.path.join("artifacts", "model.pkl")
		self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

	def predict(self, features):
		try:
			model = load_object(self.model_path)
			preprocessor = load_object(self.preprocessor_path)

			transformed_features = preprocessor.transform(features)
			predictions = model.predict(transformed_features)

			return predictions

		except Exception as e:
			raise CustomException(e, sys)
