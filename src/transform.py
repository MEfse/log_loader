import os
import pandas as pd

from logger_config import logger

class Preprocessing():
    def __init__(self):
        from extract import LoadParams, LoaderCsvFile
        self.params: LoadParams = LoadParams()
        self.loader_csv: LoaderCsvFile = LoaderCsvFile()

    def preprocess_arima(self, data: pd.DataFrame):
        '''Преобразует данные в временной ряд для ARIMA

        Args: 
            data (DataFrame): Данные полученные с базы данных 

        Returns:
            time_series (DataFrame): Возвращает временной ряд
        '''
        if data.empty or 'hour' not in data.columns:
            logger.warning(
                "Пустой датафрейм или отсутствует колонка 'hour'.")
            return None
        
        data.set_index('hour', inplace=True)

        if isinstance(data, pd.Series):
            data = data.to_frame()
            logger.info("Преобразовали Series в DataFrame.")

        return data


    #def clean_last_line(file_path: str):
        #with open(file_path, "r+", encoding="utf-8") as f:
            #lines = f.readlines()
            #if lines:  
                #lines = lines[:-1]   # удаляем последнюю строку
                #f.seek(0)
                #f.truncate()
                #f.writelines(lines)