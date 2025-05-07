from src.config.path_config import path_model, path_predict, path_logs, path_connect, path_logging, path_time_series, path_tmp
from src.loader.connect_to_db import load_data_from_db
from src.loader.loader_csv_file import load_csv_file
from src.loader.loader_model import load_model, save_model
from src.feature_engineering import get_last_time, preprocess_arima
from src.model_training import load_or_train, train_model, update_model
from src.evaluation import evaluate, predict
from src.utils import adfuller_test, plot_predict

# PIPELINE
# Получение данных из БД
data = load_data_from_db()

# Преобработка данных
data = preprocess_arima(data)

# Обучение модели
# load_or_train(data)

# Оценка модели
# predict_value = predict(path_model, path_predict)
# evaluate(path_time_series, path_predict)
# plot_predict()
