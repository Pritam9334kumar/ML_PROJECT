import os
import sys

from flask import Flask, render_template, request

from src.ml_project.exception import CustomException
from src.ml_project.logger import logging
from src.ml_project.pipeline.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_datapoint():
    try:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(data.get_data_as_data_frame())
        result = round(float(pred[0]), 2)

        return render_template("index.html", results=result)

    except Exception as e:
        logging.info("Exception occurred during prediction")
        raise CustomException(e, sys)


if __name__ == "__main__":
    logging.info("Starting the Flask application")
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)