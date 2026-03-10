# Библиотеки для моделей машинного обучения
import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import pmdarima as pm
from sklearn.model_selection import GridSearchCV

from src.training.data_loader import LoadParams, DataLoader, LoaderModel    # Настройки и параметры
from src.common.config import Settings                                      # Настройки и параметры
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

import logging                                                              # Для логирования

logger = logging.getLogger(__name__)    # Создание логгера для текущего модуля

class Strategy():
    def get_query(self, path, update_days, train_days) -> str:
        """
        Выдает запрос с временным окном в зависимости от наличия модели.

        Args:
            path (dict): Путь до модели.
            update_days (str): Начало временного отрезка загрузки данных.
            train_days (str): Конец временного отрезка загрузки данных.

        Returns:
            sql (str): Запрос для базы данных.
        """

        # Проверка на наличие модели
        if self.model_exists(path):
            sql = f'''SELECT start_time, log_count
                        FROM aggregation_by_hour
                        WHERE start_time >= now() - interval '{update_days} days'
                        ORDER BY start_time;'''
            
            logger.info(f"Обучаем за последние {update_days} дней.")
            return sql
        
        # При отсутствии модели    
        else:
            sql = f'''SELECT start_time, log_count
                        FROM aggregation_by_hour
                        WHERE start_time >= now() - interval '{train_days} days'
                        ORDER BY start_time;'''
            
            logger.info(f"Обучаем за последние {train_days} дней.")
            return sql
        
    def model_exists(self, path: str) -> bool:
        """
        Проверка на наличии модели.

        Args:
            path (dict): Путь до модели.

        Returns:
            os.path.exists (bool): Возращает True при наличии модели / False при ее отсутствии
        """
        return os.path.exists(path)


class Estimator():
    def __init__(self, fit_fn, update_fn):
        self.settings = Settings()
        self.loader_data = DataLoader()
        self.loader_model = LoaderModel()
        self.strategy = Strategy()
        self.fit_fn = fit_fn
        self.update_fn = update_fn

    def fit(self, db_params, path):
        # Получение запроса для базы данных
        query = self.strategy.get_query(path, self.settings.update_days, self.settings.train_days)

        # Загрузка данных
        data = self.loader_data.get_data_from_aggregation(db_params, query)
        if data is None or len(data) == 0:
            raise ValueError("Нет данных для обучения модели")

        # Проверка на наличие модели
        if self.strategy.model_exists(path):
            # Файл модели найден
            try:
                model, params = self.loader_model.load_model(path)

            # Неудалось загрузить модель / обучение с нуля
            except Exception as e:
                logger.warning("Не удалось загрузить модель, обучаем с нуля: %s", e, exc_info=True)
                model, params = self.fit_fn(data)
                self.loader_model.save_model((model, params), path)
                logger.info("Окно обучения: %s -> %s, n=%s, params=%s",
                            data.index.min(), data.index.max(), len(data), params)
                return model

            # Файл модели найден / Дообучение модели
            model, params = self.update_fn(data, params)
            logger.info("Файл модели найден. Дообучаем модель.")
            self.loader_model.save_model((model, params), path)
            logger.info("Окно дообучения: %s -> %s, n=%s, params=%s",
                        data.index.min(), data.index.max(), len(data), params)
            return model

        # Файл модели не найден / обучение с нуля 
        model, params = self.fit_fn(data)
        logger.info("Файл модели не найден, обучаем с нуля.")
        self.loader_model.save_model((model, params), path)
        logger.info("Окно обучения: %s -> %s, n=%s, params=%s",
                    data.index.min(), data.index.max(), len(data), params)
        return model      
            

def arima_fit(data):
    model_auto = auto_arima(
        data,
        seasonal=False,
        trace=False,
        suppress_warnings=True,
        stepwise=True,
        max_p=7,
        max_q=7
    )
    params = {"order": model_auto.order}
    model = ARIMA(data, order=params["order"]).fit()
    return model, params


def arima_update(data, params):
    if isinstance(params, tuple):
        order = params
    else:
        order = params["order"]

    model = ARIMA(data, order=order).fit()
    return model, {"order": order}
    
def _series_to_prophet_df(data):
    df = data.reset_index()
    df.columns = ["ds", "y"]
    return df


def prophet_fit(data):
    df = _series_to_prophet_df(data)

    model = Prophet()
    model.fit(df)

    params = {
        "model_type": "prophet"
    }
    return model, params

def prophet_update(data, params):
    df = _series_to_prophet_df(data)

    model = Prophet()
    model.fit(df)

    return model, params

class NaiveLastValueModel:
    def __init__(self, last_value: float):
        self.last_value = float(last_value)

    def forecast(self, steps: int = 1):
        return pd.Series([self.last_value] * steps)


def naive_fit(data):
    last_value = float(data.iloc[-1])
    model = NaiveLastValueModel(last_value=last_value)
    params = {"method": "last_value"}
    return model, params


def naive_update(data, params):
    last_value = float(data.iloc[-1])
    model = NaiveLastValueModel(last_value=last_value)
    return model, params

class SeasonalNaiveModel:
    def __init__(self, history: pd.Series, season_length: int):
        self.history = history.copy()
        self.season_length = season_length

    def forecast(self, steps: int = 1):
        values = []
        hist = self.history.tolist()

        for i in range(steps):
            idx = len(hist) - self.season_length + i
            if idx < 0:
                values.append(hist[-1])
            else:
                values.append(hist[idx])

        return pd.Series(values)


def seasonal_naive_fit(data, season_length: int = 24):
    model = SeasonalNaiveModel(history=data, season_length=season_length)
    params = {"season_length": season_length}
    return model, params


def seasonal_naive_update(data, params):
    season_length = params["season_length"]
    model = SeasonalNaiveModel(history=data, season_length=season_length)
    return model, params




def sarima_fit(data):
    model_auto = pm.auto_arima(
        data,
        seasonal=True,
        m=24,
        trace=False,
        suppress_warnings=True,
        stepwise=True,
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2
    )

    params = {
        "order": model_auto.order,
        "seasonal_order": model_auto.seasonal_order
    }

    model = SARIMAX(
        data,
        order=params["order"],
        seasonal_order=params["seasonal_order"]
    ).fit(disp=False)

    return model, params


def sarima_update(data, params):
    model = SARIMAX(
        data,
        order=params["order"],
        seasonal_order=params["seasonal_order"]
    ).fit(disp=False)

    return model, params