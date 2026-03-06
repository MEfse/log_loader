import os
import sys
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from datetime import datetime, timedelta                # Для работы с датами и временем
import psycopg2                         # Для работы с PostgreSQL
import joblib                           # Для сериализации объектов
import logging                          # Для логирования

# Настройка пути к проекту
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from src.inference.service.predictor import Evaluator
from src.common.config import LoadParams                                  # Настройки и параметры

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference.service.predictor import Evaluator
from src.common.config import LoadParams

app = FastAPI(title="ARIMA Forecast Service")

# --------------------
# Health check
# --------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------
# GET /forecast (оставляем как было)
# --------------------
@app.get("/forecast")
def get_forecast():
    try:
        params = LoadParams().get_db_params()
        ev = Evaluator()
        value = ev.predict(params)
        if isinstance(value, dict):
            return value
        return {"forecast": value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------
# POST /predict (как ты хотел)
# --------------------
class PredictResponse(BaseModel):
    forecast_start: str
    forecast_end: str
    predicted_value: float
    model_version: str

@app.post("/predict", response_model=PredictResponse)
def predict():
    try:
        params = LoadParams().get_db_params()
        ev = Evaluator()
        result = ev.predict(params)

        # Если сейчас predict возвращает только число — упакуем в dict
        if not isinstance(result, dict):
            raise HTTPException(
                status_code=500,
                detail="Evaluator.predict() must return dict with forecast_start/forecast_end/predicted_value/model_version",
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------
# POST /model/reload
# --------------------
@app.post("/model/reload")
def reload_model():
    try:
        ev = Evaluator()
        ev.reload_model()
        return {"status": "model reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))