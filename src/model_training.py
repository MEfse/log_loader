# Библиотеки для моделей машинного обучения
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

from src.config.path_config import path_model, path_predict, path_logs
from src.loader.connect_to_db import load_data_from_db
from src.loader.loader_csv_file import load_csv_file
from src.loader.loader_model import load_model, save_model
from src.feature_engineering import preprocess_arima
from src.utils import adfuller_test
from src.config.logger_config import logger


def load_or_train(data):
    '''Загружает модель, если она уже есть, иначе обучает с нуля

    Args: 
        data (DataFrame) : Данные для обучения модели


    Returns:

    '''
    logger.info("Старт load_or_train.")

    try:
        model_with_order = load_model(path_model)
        update_model(model_with_order, data)

    except FileNotFoundError:
        logger.info("Файл модели не найден, обучаем с нуля.")

        # Загружаем модель
        full_data = load_data_from_db(full_load=True)
        train_model(full_data)


def train_model(data):
    '''Функция где обучается модель

    Args: 
        data (DataFrame): Данные полученные с базы данных 

    Returns:
        model (hd5): Возвращает модель
    '''
    # logger.info("Start train_model.")
    if data is None:
        logger.info(f"Данные не загружены")

    time_series = preprocess_arima(data)

    # Дифференцируем, если нестационарен
    if not adfuller_test(time_series):
        time_series = time_series.diff().dropna()

    # Автоматический подбор параметров ARIMA
    model_auto = auto_arima(
        time_series,
        seasonal=False,
        trace=True,
        suppress_warnings=True,
        stepwise=False,   # Отключаем жадный поиск
        max_p=7, max_q=7  # Даем шанс выбрать сложнее параметры
    )

    best_p, best_d, best_q = model_auto.order

    # Обучение модели
    model = ARIMA(time_series, order=(best_p, best_d, best_q)).fit()
    save_model((model, (best_p, best_d, best_q)), path_model)

    return model


def update_model(model_with_order, data):
    '''Дообучает ARIMA-модель на новых данных

    Args: 
        data (DataFrame): Данные полученные с базы данных 

    Returns:
        model (hd5): Возвращает временной ряд
    '''

    model, order = model_with_order

    if model is None:
        logger.info("Модель не загружена")

    # Обновляем модель новыми наблюдениями
    model = ARIMA(data, order=order).fit()

    # Сохраняем обновленную модель
    save_model((model, order), path_model)
    logger.info("Модель дообучена и сохранена.")
