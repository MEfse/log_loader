import logging
import os
from dotenv import load_dotenv

# Загрузка .env файла
load_dotenv()

# Параметры для логирования
path_logging = os.getenv("PATH_LOGGING")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(path_logging, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()
