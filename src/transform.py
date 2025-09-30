import os
import pandas as pd

from src.logger_config import logger

class Preprocessing():
    def __init__(self):
        from src.extract import LoadParams
        self.params: LoadParams = LoadParams()

    def preprocess_arima(self, data: pd.DataFrame):
        '''Преобразует данные в временной ряд для ARIMA

        Args: 
            data (DataFrame): Данные полученные с базы данных 

        Returns:
            time_series (DataFrame): Возвращает временной ряд
        '''

        # Проверка на пустые данные или если timestamp нет в данных
        if data.empty or 'hour' not in data.columns:
            logger.warning(
                "Пустой датафрейм или отсутствует колонка 'hour'.")
            return None

        # Делаем индекс timestamp
        data.set_index('hour', inplace=True)

        # Проверка и преобразование в DataFrame, если это Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
            logger.info("Преобразовали Series в DataFrame.")

        return data


