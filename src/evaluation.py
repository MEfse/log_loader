# Библиотеки для анализа данных
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score
from datetime import datetime

from src.logger_config import logger

class Metrics():
    def __init__(self):
        from src.extract import LoadParams, LoaderModel, LoaderCsvFile
        self.params: LoadParams = LoadParams()
        self.loader: LoaderCsvFile = LoaderCsvFile()
        self.get_model: LoaderModel = LoaderModel()

    def predict(self):
        '''Прогноз на 1 час вперед

        Args: 
            path_model (str): Путь до обученной модели

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
            logger.info(
                f"Прогноз сохранен в {self.params.PATH_PREDICT_ARIMA}. Значение {predict_value}")

            return predict_value

        except Exception as e:
            logger.error(f"Модель не загружена, прогноз невозможен. Ошибка {e}.")
            raise


    def evaluate(self):
        '''Оценивает точность предсказания

        Args: 

        Returns:
            mae (float): Возвращает mae
        '''

        #self.loader_csv.load_predict
        real = load_csv_file(path_time_series)
        predict = load_csv_file(path_predict)

        real['hour'] = pd.to_datetime(real['hour'], errors='coerce')
        predict['hour'] = pd.to_datetime(predict['hour'], errors='coerce')

        print("real columns:", real.columns, real.info())
        print("predict columns:", predict.columns, predict.info())

        merged = pd.merge(real, predict, on='timestamp', how='inner')

        mae = abs(merged['log_level'] - merged['prediction'])
        logger.info(f"MAE: {mae}")
        return mae
