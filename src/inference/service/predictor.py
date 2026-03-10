# Библиотеки для моделей машинного обучения
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

from src.common.config import Settings, LoadParams                                  # Настройки и параметры
from src.training.data_loader import DataLoader, LoaderModel                        # Настройки и параметры

logger = logging.getLogger(__name__)   # Создание логгера для текущего модуля

class Evaluator():
    def __init__(self):
        self.params = LoadParams() 
        self.settings = Settings()
        self.loader_data = DataLoader()
        self.loader_model = LoaderModel()

    def predict(self, db_params, model_version, path, start_ts: str | None = None, end_ts: str | None = None):
        """Прогноз на 1 час вперед"""

        try:
            # Загрузка модели
            model, _ = self.loader_model.load_model(path)

            # Время прогноза
            if end_ts is None:
                base = datetime.now().replace(minute=0, second=0, microsecond=0)
            else:
                base = pd.to_datetime(end_ts).to_pydatetime()
                base = base.replace(minute=0, second=0, microsecond=0)

            forecast_start = base + timedelta(hours=1)
            forecast_end = forecast_start + timedelta(hours=1)

            # Прогноз в зависимости от модели
            # ARIMA
            if model_version == "arima":
                forecast = model.forecast(steps=1)
                predicted_value = float(forecast.iloc[0])
            
            # Prophet
            elif model_version == "prophet":
                forecast_start_naive = pd.Timestamp(forecast_start)

                if forecast_start_naive.tz is not None:
                    forecast_start_naive = forecast_start_naive.tz_localize(None)

                future = pd.DataFrame({"ds": [forecast_start_naive]})
                forecast_df = model.predict(future)
                predicted_value = float(forecast_df["yhat"].iloc[0])

            # Исключение
            else:
                raise ValueError(f"Неизвестная модель: {model_version}")

            # Запрос 
            insert_sql = f"""
                INSERT INTO forecast (
                    forecast_start_time,
                    forecast_end_time,
                    predicted_value,
                    model_version
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (forecast_start_time, model_version)
                DO UPDATE SET
                    forecast_end_time = EXCLUDED.forecast_end_time,
                    predicted_value   = EXCLUDED.predicted_value,
                    model_version     = EXCLUDED.model_version;
            """

            # Коннект до базы данных 
            with psycopg2.connect(**db_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, (forecast_start, forecast_end, predicted_value, model_version))

            logger.info("Forecast saved: %s -> %s, value=%s, model_version=%s",
                forecast_start, forecast_end, predicted_value, model_version)

            return predicted_value

        except Exception as e:
            logger.error("Forecast failed: %s", e, exc_info=True)
            raise


class MetricsLoader:
    def save_metrics_from_db(self, db_params: dict) -> None:
        sql = """
            INSERT INTO metrics_model (
                model_version,
                start_time,
                end_time,
                mae,
                mse,
                mape
            )
            SELECT
                f.model_version,
                f.forecast_start_time AS start_time,
                f.forecast_end_time   AS end_time,
                ABS(a.log_count - f.predicted_value) AS mae,
                POWER(a.log_count - f.predicted_value, 2) AS mse,
                CASE
                    WHEN a.log_count = 0 THEN NULL
                    ELSE ABS((a.log_count - f.predicted_value) / a.log_count::double precision) * 100
                END AS mape
            FROM forecast f
            JOIN aggregation_by_hour a
                ON a.start_time = f.forecast_start_time
            ON CONFLICT (model_version, start_time)
            DO UPDATE SET
                end_time = EXCLUDED.end_time,
                mae      = EXCLUDED.mae,
                mse      = EXCLUDED.mse,
                mape     = EXCLUDED.mape;
        """

        try:
            with psycopg2.connect(**db_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
            logger.info("Метрики успешно записаны в metrics_model")
        except Exception as e:
            logger.error("Ошибка записи метрик: %s", e, exc_info=True)
            raise