# Библиотеки для анализа данных
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score
from datetime import datetime

from logger_config import logger

class Evaluator():
    def __init__(self):
        from extract import LoadParams, LoaderModel, LoaderCsvFile
        self.params: LoadParams = LoadParams()
        self.loader_csv: LoaderCsvFile = LoaderCsvFile()
        self.get_model: LoaderModel = LoaderModel()

    def predict(self):
        '''Прогноз на 1 час вперед

        Args: 

        Returns:
            forecast (float): Прогнозируемое значение
        '''

        try:
            model_with_order = self.get_model.load_model(self.params.PATH_MODEL_ARIMA)
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
        '''Готовит датафрейм для оценки: hour, log_count(факт), prediction

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

