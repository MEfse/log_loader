from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from typing import Optional


#from src.extract import LoadParams
from src.logger_config import logger

#params = LoadParams()
#path_model = params.PATH_MODEL_ARIMA
#path_time_series = params.PATH_TIME_SERIES

def adfuller_test(data):
    '''Проверяет ряд на стационарность с помощью теста Дики-Фуллера

    Args: 
        data (DataFrame): Данные полученные с базы данных 

    Returns:
        time_series (DataFrame): Возвращает временной ряд
    '''
    ALPHA = 0.05
    result = adfuller(data)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')

    if result[1] < ALPHA:
        print("Ряд стационарен, можно использовать ARIMA!")
        return True
    else:
        print("Ряд нестационарен, требуется дифференцирование!")
        return False


def plot_predict():
    '''Визуализация предсказаний ARIMA

    Args: 


    Returns:

    '''

    time_series = LoadParams.load_csv_file(path_time_series)

    try:
        model = LoadParams.load_model(path_model)
        forecast_steps = 24
        forecast_obj = model.get_forecast(steps=forecast_steps)
        conf_int = forecast_obj.conf_int()

        plt.figure(figsize=(12, 6))
        plt.plot(time_series, label='Исторические данные')
        plt.plot(forecast_obj.predicted_mean, label='Прогноз',
                 linestyle='dashed', color='red')
        plt.fill_between(
            conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.3)
        plt.xlabel('Дата')
        plt.ylabel('Количество логов')
        plt.legend()
        plt.grid(True)
        plt.show()
    except:
        logger.error("Модель не загружена")


def split_data(X, y):
    '''
    Делит данные на обучающую, валидационную и тестовую выборки

    Args:
        X(DataFrame): Признаки
        y(DataFrame): Целевые метрики

    Returns:
        (X_train, y_train)(tuple): Возвращает обучающую выборку
        (X_val, y_val)(tuple): Возвращает валидационную выборку
        (X_test, y_test)(tuple): Возвращает тестовую выборку
    '''

    X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print(f'Размер обучающей выборки:     {X_train.shape[0]:<5}   {round(X_train.shape[0]/X.shape[0]*100, 2):>5}%')
    print(f'Размер валидационной выборки: {X_val.shape[0]:<5}   {round(X_val.shape[0]/X.shape[0]*100, 2):>5}%')
    print(f'Размер тестовой выборки:      {X_test.shape[0]:<5}   {round(X_test.shape[0]/X.shape[0]*100, 2):>5}%')

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
