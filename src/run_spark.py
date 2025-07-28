from pyspark.sql import SparkSession
from src.config.path_config import path_logs


def spark_connect():
    # Создаем Spark-сессию
    spark = SparkSession.builder\
        .appName("RemoteSparkApp") \
        .master("spark://10.55.6.75:7077") \
        .getOrCreate()

    data = spark.read.csv(str(path_logs), header=True)

    print("Count of rows:", data.count())  # Триггер работы с отчетом
    data.show(5)

    return data
