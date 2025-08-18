from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt

from src.path_config import path_model, path_time_series
from src.extract import load_csv_file, load_model
from src.logger_config import logger


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

# def create_or_delete_file(create: bool = True,
    # path: str = path_tmp) -> None:
    '''
    Создание временного файла

    Args: 
        create (bool): Создавать ли новый файл 

    Returns:

    '''
    # try:
    # if create == True:
    # Создание временного файла
    # save_csv_file(pd.DataFrame(), path_tmp, file_desc='Временный файл')
    # else:
    # os.remove(path)
    # logger.info(f'create_or_delete_file - Временный файл удален.')
    # except Exception as e:
    # logger.error(f'create_or_delete_file - Ошибка {e}')


# def old_load_data_from_db(full_load: bool = False,
    # path: str = path_logs) -> pd.DataFrame:
    '''Загружает данные из PostgreSQL

    Args: 
        full_load (bool): Загружать все данные или только за последний час
        path (str): Путь до файла recent_logs.csv (по умолчанию)

    Returns:
        data (DataFrame): Возвращает данные полученные из базы данных если соединение успешно или None если не удалось
    '''
    # logger.info("Старт load_data_from_db.")

    # Загружаем параметры подключения к базе данных
    # DB_PARAMS = load_connect_csv(path_inconnect)
    # if DB_PARAMS is None:
    # logger.error("Не удалось загрузить параметры подключения")
    # return pd.DataFrame()

    # Проверяем существует ли файл по пути path, если нет, то full_load = True
    # if not os.path.exists(path):
    # full_load = True

    # Загружаем исторические данные из логов recent_logs.csv
    # recent_logs = load_recent_logs(path)

    # Если full_load = False, то получаем последнее время из recent_logs.csv
    # interval_str = None
    # if not full_load:
    # interval_str = get_interval(recent_logs)
    # if interval_str is None:
    # return pd.DataFrame()

    # Подключение к базе данных
    # try:
    # query = build_query(full_load, interval_str)
    # with psycopg2.connect(**DB_PARAMS) as conn:
    # with tqdm(total=1, desc="Загрузка данных из БД", bar_format="{desc}: {elapsed}") as pbar:
    # data_from_db = pd.read_sql_query(query, conn)
    # pbar.update(1)

    # Если data_from_db пустой, то возвращаем пустой датафрейм
    # if data_from_db.empty:
    # logger.info("Нет новых данных из базы данных.")
    # return pd.DataFrame()

    # Объединяем исторические данные с новыми
    # concat_data = merge_and_sort_logs(recent_logs, data_from_db)
    # Рассчет кол-во добавленных данных
    # new_rows_added = len(concat_data) - len(recent_logs)

    # if new_rows_added > 0:
    # save_csv_file(
    # concat_data.iloc[-new_rows_added:], path_logs, append=True)
    # logger.info(
    # f"Добавлено {new_rows_added} строк. Всего: {len(concat_data)}")
    # else:
    # logger.info("Новых данных нет.")

    # return concat_data if new_rows_added > 0 else recent_logs

    # Если соединение не удалось
    # except Exception as e:
    # logger.error(f"Ошибка при соединении с БД: {e}")
    # return pd.DataFrame()
