from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt

from src.config.path_config import path_model, path_time_series
from src.loader.loader_csv_file import load_csv_file
from src.config.logger_config import logger


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

    from src.loader.loader_csv_file import load_model

    time_series = load_csv_file(path_time_series)

    try:
        model = load_model(path_model)
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


# def split_data(X, y):
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

    # X_train, X_temp, y_train, y_temp = train_test_split(
    # X, y, test_size=0.4, stratify=y, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(
    # X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # print(
    # f'Размер обучающей выборки:     {X_train.shape[0]:<5}   {round(X_train.shape[0]/X.shape[0]*100, 2):>5}%')
    # print(
    # f'Размер валидационной выборки: {X_val.shape[0]:<5}   {round(X_val.shape[0]/X.shape[0]*100, 2):>5}%')
    # print(
    # f'Размер тестовой выборки:      {X_test.shape[0]:<5}   {round(X_test.shape[0]/X.shape[0]*100, 2):>5}%')

    # return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# def preprocess_data(data):
    '''
    Обрабатывает данные перед обучением модели

    Args:
        data(DataFrame): Данные на вход, которые нужно преобработать

    Returns:
        data(DataFrame): Возвращает преобработанные данные
    '''

    # data = data.drop_duplicates().reset_index(drop=True)
    # logger.info("Удалены дубликаты.")
    # data = data.dropna(subset=['timestamp'])
    # logger.info("Удаляем строки, где `timestamp` не распарсился.")
    # vectorizer = TfidfVectorizer(
    # stop_words='english', max_features=5000, max_df=0.9, min_df=5)
    # X = vectorizer.fit_transform(data['message'].astype(str))
    # logger.info("Проведена TF-IDF векторизация.")
    # encoder = LabelEncoder()
    # y = encoder.fit_transform(data['log_level'])
    # logger.info("Проведено кодирование целевого признака.")

    # return X, y


# def load_existing_logs(path: str) -> pd.DataFrame:
    '''Загрузка последнего времени

    Args: 
        path (str): Путь до файла

    Returns:
        (str): Возвращаем пустой датафрейм
         (pd.DataFrame): Возвращаем пустой датафрейм
    '''
    # try:
    # if os.path.exists(path):
    # return load_recent_logs(path)
    # else:
    # logger.info("Файл логов не найден. Начинаем с пустого лога.")
    # return pd.DataFrame()
    # except Exception as e:
    # logger.error(f"Ошибка при загрузке предыдущих логов: {e}")
    # return pd.DataFrame()


# def get_interval(path: str) -> Optional[str]:
    '''Загрузка последнего времени 

    Args: 
        path (str): Передаем данные для в записи в connect.csv

    Returns:

    '''
    # Получаем последнее время
    # last_time = get_last_time(path)

    # Если последнее время пустое
    # if last_time is None:
    # logger.error("Не удалось получить последнее время.")
    # return None
    # try:
    # last_time_obj = datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S')
    # time_diff = datetime.now() - last_time_obj
    # interval = f"{max(1, int(time_diff.total_seconds() / 60))} minutes"
    # logger.info(f"Интервал загрузки: {interval}")
    # return interval
    # except Exception as e:
    # logger.error(f"Ошибка при расчёте интервала: {e}")
    # return None
