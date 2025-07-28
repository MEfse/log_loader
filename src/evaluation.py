# Библиотеки для анализа данных
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score
from datetime import datetime

from src.loader.loader_csv_file import load_csv_file
from src.loader.loader_model import load_model
# from src.feature_engineering import preprocess_arima
from src.config.logger_config import logger


def predict(path_model, path_predict):
    '''Прогноз на 1 час вперед

    Args: 
        path_model (str): Путь до обученной модели

    Returns:
        forecast (float): Прогнозируемое значение
    '''
    logger.info("Старт predict.")

    try:
        model_with_order = load_model(path_model)
        model, _ = model_with_order

        predict = model.forecast(steps=1)
        predict_value = predict.iloc[0]

        # Округление до часа
        now = datetime.now().replace(minute=0, second=0, microsecond=0)

        predict_df = pd.DataFrame(
            {'timestamp': [now],
             'prediction': [predict_value]})

        if not os.path.exists(path_predict):
            predict_df.to_csv(path_predict, mode='a',
                              date_format='%Y-%m-%d %H:%M:%S', index=False)
        else:
            predict_df.to_csv(path_predict, mode='a',
                              date_format='%Y-%m-%d %H:%M:%S', header=False, index=False)
        logger.info(
            f"Прогноз сохранен в {path_predict}. Значение {predict_value}")

        return predict_value

    except Exception as e:
        logger.error(f"Модель не загружена, прогноз невозможен. Ошибка {e}.")
        return None


def evaluate(path_time_series, path_predict):
    '''Оценивает точность предсказания

    Args: 

    Returns:
        mae (float): Возвращает mae
    '''
    logger.info("Старт evaluate.")

    real = load_csv_file(path_time_series)
    predict = load_csv_file(path_predict)

    real['timestamp'] = pd.to_datetime(real['timestamp'], errors='coerce')
    predict['timestamp'] = pd.to_datetime(
        predict['timestamp'], errors='coerce')

    print("real columns:", real.columns, real.info())
    print("predict columns:", predict.columns, predict.info())

    merged = pd.merge(real, predict, on='timestamp', how='inner')

    mae = abs(merged['log_level'] - merged['prediction'])
    logger.info(f"MAE: {mae}")
    return mae
