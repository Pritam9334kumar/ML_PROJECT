# End to End Data Science Project

## What this project does

This project trains a student performance regression model, saves the trained model and preprocessor in `artifacts/`, and serves predictions through a Flask app.

## Before deploying

1. Activate the virtual environment.
2. Run training first: `python main.py` or `.\.venv\Scripts\python.exe main.py` on Windows.
3. Confirm these files exist:
   - `artifacts/model.pkl`
   - `artifacts/preprocessor.pkl`
4. Add a `.env` file if you use MySQL, DagsHub, or MLflow credentials.
5. Test the Flask app locally with `python app.py`.
6. If you deploy with Docker, rebuild the image after every code or model update.

## Run locally

```bash
python main.py
python app.py
```

On Windows with this repo setup, use:

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe app.py
```

## Docker

The Dockerfile runs the Flask app with `gunicorn` on port `5000`.