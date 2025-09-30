from pyspark.sql import SparkSession
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
