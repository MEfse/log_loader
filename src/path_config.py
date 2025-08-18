from pathlib import Path

# BASE_DIR = Path.home() / "Desktop/project" # Windows
BASE_DIR = Path("/home/user/project") # Ubuntu

path_model = BASE_DIR / "models/model.hd5"
path_predict = BASE_DIR / "data/ARIMA/predict_arima.csv"
path_logs = BASE_DIR / "data/ARIMA/recent_logs.csv"
path_exconnect = BASE_DIR / "data/external_connection.csv"
path_inconnect = BASE_DIR / "data/internal_connection.csv"
path_logging = BASE_DIR / "logging.log"
path_time_series = BASE_DIR / "data/ARIMA/time_series.csv"
path_tmp = BASE_DIR / "data/ARIMA/data_tmp.csv"
