#Главный скрипт обучения:

#взять новые данные

#preprocess (через common)

#обучить модель

#сохранить артефакт во временное место


# Библиотеки для моделей машинного обучения
import os
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.model_selection import GridSearchCV

from src.training.data_loader import LoadParams, DataLoader, LoaderModel    # Настройки и параметры
from src.common.config import Settings, LoadParams                          # Настройки и параметры
from statsmodels.tsa.arima.model import ARIMA

import logging                                                              # Для логирования

#from utils import adfuller_test
logger = logging.getLogger(__name__)    # Создание логгера для текущего модуля

class TrainModel():
    def __init__(self):
        self.params = LoadParams() 
        self.settings = Settings()
        self.loader_data = DataLoader()
        self.loader_model = LoaderModel()

    def load_or_train(self, DB_PARAMS):
        '''Загружает модель, если она уже есть, иначе обучает с нуля

        Args: 
            data (DataFrame) : Данные для обучения модели

        Returns:

        '''
        #DB_PARAMS = self.params.get_db_params()

        # Путь до модели
        path = self.settings.path_model_arima

        # Проверка на наличие модели
        if os.path.exists(path):
            sql = '''SELECT start_time, log_count
                        FROM aggregation_by_hour
                        ORDER BY start_time DESC
                        LIMIT 1;'''
            
            logger.info("Файл найден.")

            model_with_order = self.loader_model.load_model(path)                   # Загрузка модели
            data = self.loader_data.get_data_from_aggregation(DB_PARAMS, sql)       # Загрузка данных
            self.update_model(model_with_order, data)                               # Апдейт модели
        else:
            sql = '''SELECT start_time, log_count
                        FROM aggregation_by_hour
                        WHERE start_time >= now() - interval '30 days'
                        ORDER BY start_time;'''
            
            logger.warning("Файл модели не найден, обучаем с нуля.")

            data = self.loader_data.get_data_from_aggregation(DB_PARAMS, sql)       # Загрузка данных
            self.train_model(data)                                                  # Обучение с нуля


    def train_model(self, data):
        '''Функция где обучается модель

        Args: 
            data (DataFrame): Данные полученные с базы данных 

        Returns:
            model (hd5): Возвращает модель
        '''
        if data is None:
            logger.info(f"Данные не загружены")

        # Дифференцируем, если нестационарен
        #if not adfuller_test(time_series):
            #time_series = time_series.diff().dropna()

        # Автоматический подбор параметров ARIMA
        model_auto = auto_arima(
            data,
            seasonal=False,
            trace=True,
            suppress_warnings=True,
            stepwise=False,             # Отключаем жадный поиск
            max_p=7, max_q=7            # Даем шанс выбрать сложнее параметры
        )

        best_p, best_d, best_q = model_auto.order

        # Обучение модели
        model = ARIMA(data, order=(best_p, best_d, best_q)).fit()
        self.loader_model.save_model((model, (best_p, best_d, best_q)), self.settings.path_model_arima)

        return model


    def update_model(self, model_with_order, data):
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
        model = ARIMA(data, order=order, freq="H").fit()

        # Сохраняем обновленную модель
        self.get_model.save_model((model, order), self.params.PATH_MODEL_ARIMA)
        logger.info("Модель дообучена и сохранена.")
