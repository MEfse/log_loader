from pathlib import Path

BASE_DIR = Path.home() / "Desktop/project"

path_model = BASE_DIR / "models/model.hd5"
path_predict = BASE_DIR / "data/ARIMA/predict_arima.csv"
path_logs = BASE_DIR / "data/ARIMA/recent_logs.csv"
path_connect = BASE_DIR / "data/connect.csv"
path_logging = BASE_DIR / "logging.log"
path_time_series = BASE_DIR / "data/ARIMA/time_series.csv"
path_tmp = BASE_DIR / '/data/ARIMA/data_tmp.tmp'
