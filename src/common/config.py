import os
import logging
from dotenv import load_dotenv
import psycopg2
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

logger = logging.getLogger(__name__)

# Загружаем параметры из .env
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    log_level: str = "INFO"
    path_logging: str | None = None

    model_name: str = "my_model"
    registry_path: str = "./model_registry"

    inference_host: str = "0.0.0.0"
    inference_port: int = 8000

    min_mse : float = 1000.0

    # Параметры БД
    db_name: str = os.getenv("DB_NAME")
    db_user: str = os.getenv("DB_USER")
    db_password: str = os.getenv("DB_PASSWORD")
    db_ip: str = os.getenv("DB_INCONN")
    db_port: str = os.getenv("DB_PORT")

    # Пути до файлов
    path_model_arima: str = os.getenv("PATH_MODEL_ARIMA")
    path_predict_arima: str = os.getenv("PATH_PREDICT_ARIMA")
    path_logs: str = os.getenv("PATH_LOGS")
    path_logging: str = os.getenv("PATH_LOGGING")
    path_time_series: str = os.getenv("PATH_TIME_SERIES")

    DB_PARAMS: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DB_PARAMS = {
            'dbname': self.db_name,
            'user': self.db_user,
            'password': self.db_password,
            'host': self.db_ip,
            'port': self.db_port
        }

class LoadParams:
    def __init__(self):
        try:
            # загружаем параметры через Pydantic Settings
            self.settings = Settings()
            logger.info("Параметры конфигурации загружены.")
        except Exception as e:
            logger.error(f'Ошибка загрузки параметров: {e}')
            raise
    
    def get_db_params(self):
        """
        Получает параметры подключения к базе данных с обработкой ошибок.
        """
        try:
            # Подключение
            with psycopg2.connect(**self.settings.DB_PARAMS, connect_timeout=2) as conn:
                logger.info("Подключение успешно.")
                return self.settings.DB_PARAMS

        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            raise
