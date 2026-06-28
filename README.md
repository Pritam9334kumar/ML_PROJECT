# End to End Student Performance Prediction

This project trains a regression model to predict a student's math score from demographic and study features, then serves predictions through a Flask app.

## Project Structure

- `main.py` - training entrypoint
- `app.py` - Flask app for prediction
- `artifacts/` - saved model and preprocessor files
- `src/ml_project/` - project source code
- `templates/` - HTML templates
- `static/` - CSS and other static assets

## Before Deploying

1. Activate the virtual environment.
2. Train the model first:
   - Windows: `.\.venv\Scripts\python.exe main.py`
   - Mac/Linux: `python main.py`
3. Confirm these files exist:
   - `artifacts/model.pkl`
   - `artifacts/preprocessor.pkl`
4. Make sure your `.env` file has any required MySQL, DagsHub, or MLflow credentials.
5. Test the Flask app locally:
   - Windows: `.\.venv\Scripts\python.exe app.py`
   - Mac/Linux: `python app.py`
6. If you use Docker, rebuild the image after each code or model update.

## Deploy on Render

1. Push this project to GitHub.
2. Create a new Render Web Service from the repository.
3. Use the following settings:
   - Environment: `Python`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Make sure the repository includes:
   - `app.py`
   - `requirements.txt`
   - `render.yaml`
   - `templates/`
   - `static/`
   - `artifacts/model.pkl`
   - `artifacts/preprocessor.pkl`
5. Deploy the service and open the Render URL.
6. If you retrain the model, commit the new artifacts and redeploy.

## Run Locally

```bash
python main.py
python app.py
```

## Docker Run

The Dockerfile runs the Flask app with `gunicorn` on port `5000`.

```bash
docker build -t student-performance-app .
docker run -p 5000:5000 student-performance-app
```

## Notes

- The training pipeline uses the saved preprocessing object from `artifacts/preprocessor.pkl`.
- The prediction app expects the trained model file at `artifacts/model.pkl`.
- The current app is designed for local deployment or container deployment.
