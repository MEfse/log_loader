import os
from dotenv import load_dotenv

# Загружаем .env
load_dotenv()

# Получаем переменные
db_user = os.getenv("LOG_LOADER_DB_USER")
db_password = os.getenv("LOG_LOADER_DB_PASSWORD")

print(f"User: {db_user}, Password: {db_password}")