# Библиотеки для моделей машинного обучения
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

from utils import adfuller_test
from logger_config import logger

class TrainModel():
    def __init__(self):
        from extract import LoadParams, LoaderCsvFile, LoaderModel
        from transform import Preprocessing
        self.params: LoadParams = LoadParams()
        self.loader: LoaderCsvFile = LoaderCsvFile()
        self.get_model: LoaderModel = LoaderModel()
        self.preprocessing: Preprocessing = Preprocessing()

    def load_or_train(self, data):
        '''Загружает модель, если она уже есть, иначе обучает с нуля

        Args: 
            data (DataFrame) : Данные для обучения модели


        Returns:

        '''
        #data = self.loader.load_recent_logs(self.params.PATH_LOGS)
        data_pre = self.preprocessing.preprocess_arima(data)
        try:
            model_with_order = self.get_model.load_model(self.params.PATH_MODEL_ARIMA)
            self.update_model(model_with_order, data_pre)
        except FileNotFoundError:
            logger.info("Файл модели не найден, обучаем с нуля.")
            self.train_model(data_pre)


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
            stepwise=False,   # Отключаем жадный поиск
            max_p=7, max_q=7  # Даем шанс выбрать сложнее параметры
        )

        best_p, best_d, best_q = model_auto.order

        # Обучение модели
        model = ARIMA(data, order=(best_p, best_d, best_q), freq="H").fit()
        self.get_model.save_model((model, (best_p, best_d, best_q)), self.params.PATH_MODEL_ARIMA)

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
