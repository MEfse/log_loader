import os
import pandas as pd
import psycopg2
import joblib
#from tqdm import tqdm
from datetime import datetime
from typing import Optional, List, Any
from dotenv import load_dotenv

from src.logger_config import logger
from src.forecast import TrainModel
#from src.transform import Preprocessing
from src.evaluation import Metrics

class Run:
    def __init__(self):
        self.params = LoadParams()
        self.loader_data = LogDataLoader()
        self.loader_csv = LoaderCsvFile()
        #self.preprocessing = Preprocessing()
        self.loader_model = TrainModel()
        self.evaluation = Metrics()
        self.DB_PARAMS = None
        self.data = None

    def execute(self):
        # Загружаем параметры подключения
        try:
            self.DB_PARAMS = self.load_params()             
        except Exception as e:
            logger.exception("Шаг load_params провалился: %s", e)
            return

        # Загружаем данные из БД
        try:
            self.data = self.get_data_db(self.DB_PARAMS)             
        except Exception as e:
            logger.exception("Шаг get_data_db провалился: %s", e)
            return

        # Обучаем модель      
        try:
            self.load_model(self.data)          
        except Exception as e:
            logger.exception("Шаг load_model провалился: %s", e)
            return    
        
        # Предсказание     
        try:
            self.predict_arima()          
        except Exception as e:
            logger.exception("Шаг predict_arima провалился: %s", e)
            return  
                          
    def load_params(self):
        self.DB_PARAMS = self.params.get_db_params()
        return self.DB_PARAMS

    def get_data_db(self, DB_PARAMS):
        data_list = []
        count = 0
        for data_chunk in self.loader_data.get_data(DB_PARAMS):
            count += len(data_chunk)
            logger.info(f'Получено {count} данных.') 
            self.loader_csv.save_csv_file(data_chunk, self.params.PATH_LOGS)
            data_list.append(data_chunk)
        
        data = pd.concat(data_list)
        return data

    def load_model(self, data):
        self.loader_model.load_or_train(data)

    def predict_arima(self):
        self.evaluation.predict()
      
class LoadParams:
    def __init__(self):
        try:          
            load_dotenv()
            #logger.info(f'Параметры подключения загружены.') 
        except Exception as e:
            logger.error(f'Ошибка. Отсутствует файл .env {e}')
            raise

        self.db_name = os.getenv("LOG_LOADER_DB_NAME") 
        self.db_user = os.getenv("LOG_LOADER_DB_USER") 
        self.db_password = os.getenv("LOG_LOADER_DB_PASSWORD")
        self.db_exconn = os.getenv("LOG_LOADER_DB_EXCONN") 
        self.db_inconn = os.getenv("LOG_LOADER_DB_INCONN") 
        self.db_port = os.getenv("LOG_LOADER_DB_PORT") 

        self.PATH_MODEL_ARIMA = os.getenv("PATH_MODEL_ARIMA") 
        self.PATH_PREDICT_ARIMA = os.getenv("PATH_PREDICT_ARIMA") 
        self.PATH_LOGS = os.getenv("PATH_LOGS")
        self.PATH_LOGGING = os.getenv("PATH_LOGGING") 
        self.PATH_TIME_SERIES = os.getenv("PATH_TIME_SERIES") 

        self.DB_PARAMS = {
            'dbname': self.db_name, 
            'user': self.db_user, 
            'password': self.db_password,
            'host': self.db_inconn, 
            'port': self.db_port
        }
    
    def get_db_params(self):
        '''
        Загрузка параметров подключения (по внутреннему или внешнему IP)

        Args: 

        Returns:
            DB_PARAMS(dict): Параметры подключения к БД
        '''

        try:
            with psycopg2.connect(**self.DB_PARAMS, connect_timeout=2) as conn:
                logger.info(f"Подключение по локальной сети.")
                return self.DB_PARAMS
        except psycopg2.OperationalError:
            self.DB_PARAMS['host'] = self.db_exconn
            logger.info(f"Подключение по внешней сети.")
            return self.DB_PARAMS
        except Exception as e:
            logger.error(f"Неизвестная ошибка при подключении: {e}")
           
class LogDataLoader:
    def __init__(self):
        self.batch_size = 1000
        self.params = LoadParams()                          
        self.loader = LoaderCsvFile()
        self.get_query = GetInterval()

    def get_data(self, DB_PARAMS):
        try:
            with psycopg2.connect(**DB_PARAMS) as conn:
                with conn.cursor(name='batched_cursor') as cursor:
                    if not os.path.exists(self.params.PATH_LOGS):
                        logger.info(f"Загружаем все данные из БД.")
                        cursor.execute(f"""SELECT DATE_TRUNC('hour', timestamp) AS time, COUNT(*) AS log_count
                                            FROM logs
                                            GROUP BY time
                                            ORDER BY time;""")
                    else:
                        data = self.loader.load_recent_logs(self.params.PATH_LOGS)
                        interval_str = self.get_query.get_interval(data)
                        cursor.execute(f"""SELECT DATE_TRUNC('hour', timestamp) AS time, COUNT(*) AS log_count
                                            FROM logs
                                            WHERE timestamp >= NOW() - INTERVAL '{interval_str}'
                                            GROUP BY time
                                            ORDER BY time;""")

                    while True:
                        rows = cursor.fetchmany(self.batch_size)
                        if not rows:
                            break
                        chunk_df = pd.DataFrame(rows, columns=['hour', 'log_count'])
                        yield chunk_df  

        except Exception as e:
            logger.error(f"Ошибка при соединении с БД: {e}")
            raise
        
class GetInterval:
    def __init__(self):
        self.params = LoadParams()                          
        self.loader = LoaderCsvFile()

    def get_last_time(self, data: pd.DataFrame) -> Optional[pd.Timestamp]:
        '''Загружает последннюю дату из файла recent_logs

        Args: 
            path_logs (str): Передаем путь для подключения к файлу recent_logs
        Returns:
            last_time (datetime): Возвращаем последнее время в файле recent_logs
        '''
        if data is None or data.empty:
            logger.error("Данных нет.")
            raise
        
        if 'hour' not in data.columns:
            logger.error("Колонка 'hour' отсутствует.")
            raise


        last_time = data['hour'].max()

        if pd.isna(last_time):
            logger.error("Не удалось извлечь дату.")
            raise

        logger.info(f"Получаем последнее время: {last_time}")
        return last_time
    
    def get_interval(self, data: pd.DataFrame) -> Optional[str]:
        '''
        Получение интервала времени для запроса SQL 

        Args: 
            data (pd.DataFrame): Данные по которым считается интервал времени

        Returns:
            interval (str): Интервал недостающих данных
        '''
        last_time = self.get_last_time(data)

        try:
            time_diff = datetime.now() - last_time
            interval = f"{max(1, int(time_diff.total_seconds() / 60))} minutes"
            logger.info(f"Интервал загрузки: {interval}")
            return interval
        except Exception as e:
            logger.error(f"Ошибка при расчёте интервала: {e}")
            raise               
                
class LoaderCsvFile:
    #def __init__(self):
        #pass
    
    def load_csv_file(self, 
                    path: str, 
                    parse_dates: Optional[List[str]] = None,
                    expected_columns: Optional[List[str]] = None) -> pd.DataFrame:
        '''
        Универсальная функция загрузки CSV-файлов с логированием и обработкой ошибок.

        Args:
            path (str): Путь к CSV файлу
            parse_dates (list[str], optional): Какие колонки парсить как даты
            expected_columns (list[str], optional): Если указано — создаётся пустой DataFrame с этими колонками в случае ошибки

        Returns:
            (DataFrame): Загруженный DataFrame или пустой DataFrame с нужными колонками
        '''
        try:
            data = pd.read_csv(path, parse_dates=parse_dates)
            logger.info(
                f'Файл {path} загружен.')
            return data
        except FileNotFoundError:
            logger.warning(
                f'Файл {path} не найден.')
            raise
        except pd.errors.EmptyDataError:
            logger.warning(
                f'Файл {path} пуст.')
            return pd.DataFrame(columns=expected_columns)
        except Exception as e:
            logger.error(f'Ошибка при загрузке {path}: {e}')
            raise

    def save_csv_file(self,
                    data: pd.DataFrame,
                    path: str) -> None:
        '''
        Универсальная функция сохранения DataFrame в CSV, с добавлением новых данных.

        Args:
            data (DataFrame): Данные для сохранения
            path (str): Путь к файлу

        Returns:

        '''
        # Обработка исключения, если data не DataFrame
        if not isinstance(data, pd.DataFrame):
            logger.error(f"Файл {path} не является DataFrame.")
            raise

        # Сохраняем индекс, data.index не временной
        save_index = not isinstance(data.index, pd.RangeIndex)

        if os.path.exists(path):
            mode = 'a' 
            header = False
        else:
            mode = 'w'
            header = True

        try:
            data.to_csv(path, mode=mode, header=header, index=save_index)

        except Exception as e:
            logger.error(f"save_csv_file - Ошибка при сохранении файла {path}: {e}")
            raise


    def load_recent_logs(self, path):
        '''
        Функция загрузки файла recent_logs.csv

        Args:
            path (str): Путь к CSV файлу recent_logs.csv

        Returns:
            DataFrame: Загруженный DataFrame или пустой DataFrame с нужными колонками
        '''
        return self.load_csv_file(path,
                            parse_dates=['hour'],
                            expected_columns=['hour', 'log_count'])


    def load_predict(self, path):
        '''
        Функция загрузки файла predict_arima.csv

        Args:
            path (str): Путь к CSV файлу

        Returns:
            DataFrame: Загруженный DataFrame или пустой DataFrame с нужными колонками
        '''
        return self.load_csv_file(path,
                            parse_dates=['hour'],
                            expected_columns=['hour', 'prediction'])

class LoaderModel:
    def __init__(self):
        self.params = LoadParams()                          

    def load_model(self, path):
        '''Загружает модель ARIMA

        Args: 
            path_model (str): Передаем путь для подключения к файлу модели models.hd5

        Returns:
            predict (DataFrame): Загружаем модель ARIMA
        '''
        try:
            # Загружаем обученную модель
            return joblib.load(path)
        except FileNotFoundError:
            logger.warning("Файл модели не найден.")
            raise


    def save_model(self, model_with_order: Any, path: str) -> Any:
        '''Функция для сохранения модели

        Args:
            model (hd5): Модель 
            path_model (str): Передаем путь для подключения к файлу модели models.hd5

        Returns:
            model (hd5): Сохраняем модель 
        '''
        if model_with_order:
            joblib.dump(model_with_order, path)
            logger.info("Файл модели сохранен.")
            return model_with_order
        else:
            logger.error("Модель не удалось сохранить.")
            raise
