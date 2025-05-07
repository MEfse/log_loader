import logging
from src.config.path_config import path_logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(path_logging, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()
