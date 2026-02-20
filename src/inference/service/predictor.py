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
from src.training.data_loader import LoadParams, DataLoader, LoaderModel            # Настройки и параметры
from src.common.preprocessing.pipeline import preprocess, GetInterval               # Функции для обработки данных
#from src.training.train import TrainModel                                          # Модель для обучения

logger = logging.getLogger(__name__)   # Создание логгера для текущего модуля

class Evaluator():
    def __init__(self):
        self.params = LoadParams() 
        self.settings = Settings()
        self.loader_data = DataLoader()
        self.loader_model = LoaderModel()

    def predict(self):
        '''Прогноз на 1 час вперед

        Args: 

        Returns:
            forecast (float): Прогнозируемое значение
        '''

        try:
            model_with_order = self.loader_model.load_model(self.settings.path_model_arima)
            model, _ = model_with_order

            predict = model.forecast(steps=1)
            predict_value = predict.iloc[0]

            # Округление до часа
            now = datetime.now().replace(minute=0, second=0, microsecond=0)

            predict_df = pd.DataFrame(
                {'hour': [now],
                'prediction': [predict_value]})

            if not os.path.exists(self.params.PATH_PREDICT_ARIMA):
                predict_df.to_csv(self.params.PATH_PREDICT_ARIMA, mode='a',
                                date_format='%Y-%m-%d %H:%M:%S', index=False)
            else:
                predict_df.to_csv(self.params.PATH_PREDICT_ARIMA, mode='a',
                                date_format='%Y-%m-%d %H:%M:%S', header=False, index=False)
            logger.info(f"Прогноз сохранен в {self.params.PATH_PREDICT_ARIMA}. Значение {predict_value}")

            return predict_value

        except Exception as e:
            logger.error(f"Модель не загружена, прогноз невозможен. Ошибка {e}.")
            raise

    def create_evaluate(self):
        '''Готовит датафрейм для оценки: hour, log_count, prediction

        Args: 

        Returns:
            forecast (float): Прогнозируемое значение
        '''
        real = self.loader_csv.load_recent_logs(self.params.PATH_LOGS)
        predict = self.loader_csv.load_predict(self.params.PATH_PREDICT_ARIMA)

        data = pd.merge(real, predict, on='hour', how='inner')

        if data.empty:
            logger.warning("После объединения факт/прогноз данных нет.")
        else:
            logger.info("Объединили %d точек (с %s по %s).",
                             len(data), data['hour'].min(), data['hour'].max())
        return data
    
    def evaluate_metrics(self, data: pd.DataFrame):
        '''Возвращает словарь метрик и data с колонками ошибок

        Args: 

        Returns:
            
        '''

        if data.empty:
            return {"mae": np.nan, "mse": np.nan, "rmse": np.nan}, data
        
        data = data.copy()
        err = data['log_count'] - data['prediction']
        data['abs_err'] = err.abs()
        data['sq_err'] = err.pow(2)

        mae = data['abs_err'].mean()
        mse = data['sq_err'].mean()
        rmse = np.sqrt(mse)

        metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "points": int(len(data))}#,
            #"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        #}

        return metrics, data

    def save_evaluation():
        col = ['hour', 'prediction', 'abs_err', 'sq_err']