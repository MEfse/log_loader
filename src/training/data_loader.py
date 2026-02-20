# Стандартные библиотеки
import os                                                               # Для работы с путями и директориями
import sys                                                              # Для работы с путями поиска модулей
import pandas as pd                                                     # Для работы с данными (DataFrame)
import psycopg2                                                         # Для работы с PostgreSQL
import joblib                                                           # Для сериализации объектов
from datetime import datetime                                           # Для работы с датой и временем
from typing import Optional, List, Any                                  # Для аннотаций типов
import logging                                                          # Для логирования
from pyspark.sql import SparkSession

# Настройка пути к проекту
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.common.config import Settings, LoadParams                      # Настройки и параметры
from src.common.preprocessing.pipeline import preprocess, GetInterval   # Функции для обработки данных
#from src.training.train import TrainModel                              # Модель для обучения
from src.inference.service.predictor import Evaluator                   # Оценщик для инференса

logger = logging.getLogger(__name__)                                    # Создание логгера для текущего модуля

#--------------Генератор загрузки данных---------------------------------------------------------------------
class GeneratorDataLoader:
    def __init__(self):
        self.batch_size = 10000
        self.params = LoadParams()  

    def get_data(self, sql: str | None = None, start_ts: str | None = None, end_ts: str | None = None):
        '''
        Генератор загрузки данных из БД.

        Args:
            start_ts (str): Начало временного отрезка загрузки данных.
            end_ts (str): Конец временного отрезка загрузки данных.

        Returns:
            (DataFrame): Чанк данных.
        '''

        # Используем параметры из LoadParams для подключения
        DB_PARAMS = self.params.get_db_params()  

        # Запрос в БД 
        default_sql  = """SELECT date_trunc('hour', timestamp) AS start_time,
                            date_trunc('hour', timestamp) + INTERVAL '1 hour' AS end_time,
                            COUNT(*) AS log_count
                        FROM logs 
                        WHERE timestamp >= %(start)s
                        AND timestamp < %(end)s
                        GROUP BY start_time, end_time
                        ORDER BY start_time;"""
        
        query = sql if sql is not None else default_sql

        try:
            # Создаем соединение с БД
            with psycopg2.connect(**DB_PARAMS) as conn:                  
                with conn.cursor(name='batched_cursor') as cursor:   
                    
                    cursor.execute(query, {"start": start_ts, "end": end_ts}) 

                    while True:
                        rows = cursor.fetchmany(self.batch_size)        # Батч-данных
                        if not rows:                                    # Если нет данных выход из цикла                                  
                            break
                        chunk_df = pd.DataFrame(rows, columns=['start_time', 'end_time', 'log_count']) # Формирование датафрейма с данными
                        yield chunk_df

        # При неудачном соединение с БД
        except Exception as e:
            logger.error(f"Ошибка при соединении с БД: {e}.")
            raise

#--------------Загрузка из БД---------------------------------------------------------------------
class DataLoader:
    def __init__(self):
        self.loader_data = GeneratorDataLoader()

    def get_data_from_db(self, DB_PARAMS, start_ts, end_ts):
        """
        Загружает данные из БД с учетом временных рамок и записывает их в таблицу агрегатов.

        Args:
            DB_PARAMS (dict): Параметры для подключения к базе данных.
            start_ts (str): Начало временного отрезка загрузки данных.
            end_ts (str): Конец временного отрезка загрузки данных.

        Returns:
            None: Записываем данные агрегатов в БД.
        """
        total_logs = 0  
        total_rows = 0
        upsert_sql = """INSERT INTO aggregation_by_hour (start_time, end_time, log_count)
                                        VALUES (%s, %s, %s)
                                        ON CONFLICT (start_time) 
                                        DO UPDATE SET log_count = EXCLUDED.log_count;"""
        try:
            # Создаем соединение с БД
            with psycopg2.connect(**DB_PARAMS) as conn:                 
                with conn.cursor() as cursor:                       

                    for data_chunk in self.loader_data.get_data(start_ts=start_ts, end_ts=end_ts):
                        if data_chunk is None or data_chunk.empty:
                            logger.info("Нет данных за окно %s - %s", start_ts, end_ts)
                            continue

                        # Метрики
                        chunk_logs = int(data_chunk["log_count"].sum())
                        total_logs += chunk_logs
                        total_rows += len(data_chunk) 

                        # Подготовка данных для вставки
                        values = list(zip(data_chunk["start_time"].tolist(), data_chunk["end_time"].tolist(), data_chunk["log_count"].tolist()))

                        # Выполнение операции вставки или обновления
                        cursor.executemany(upsert_sql, values)

                        # Логирование прогресса
                        logger.info("Записано/обновлено %s строк (часов) за чанк; логов в чанке: %s; всего логов: %s", 
                                        len(values), chunk_logs, total_logs)

                # Зафиксировать изменения после обработки всех чанков
                conn.commit()

                # Логируем финальный результат
                logger.info("Готово. Окно %s - %s: обновлено часов: %s, логов: %s",
                                start_ts, end_ts, total_rows, total_logs)
        except Exception as e:
            logger.error(f"Ошибка при чтении/записи агрегатов: {e}")
            raise

    def get_data_from_aggregation(self, DB_PARAMS, sql):
        """
        Загружает данные из БД с учетом временных рамок и записывает их в таблицу агрегатов.

        Args:
            DB_PARAMS (dict): Параметры для подключения к базе данных.
            start_ts (str): Начало временного отрезка загрузки данных.
            end_ts (str): Конец временного отрезка загрузки данных.

        Returns:
            None: Записываем данные агрегатов в БД.
        """

        try:
            with psycopg2.connect(**DB_PARAMS) as conn:
                with conn.cursor() as cursor:

                    cursor.execute(sql)
                    rows = cursor.fetchall()
                    df = pd.DataFrame(rows, columns=["start_time", "log_count"])
                    #print(df.head(5))
                    #print(df.info())
                    return df.set_index("start_time")["log_count"]
                
        except Exception as e:
            logger.error(f"Ошибка при чтении aggregation_by_hour: {e}")
            raise

#--------------Загрузка модели--------------------------------------------------------------------- 
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

#--------------Spark---------------------------------------------------------------------
#from src.path_config import path_logs
#from src.extract import LoadParams

#params = LoadParams()
#path_model = params.PATH_MODEL_ARIMA


def spark_connect():
    # Создаем Spark-сессию
    spark = SparkSession.builder\
        .appName("RemoteSparkApp") \
        .master("spark://10.55.6.75:7077") \
        .getOrCreate()

    #data = spark.read.csv(str(path_logs), header=True)

    #print("Count of rows:", data.count())  # Триггер работы с отчетом
    #data.show(5)

    #return data
