# Log Loader Project

## Описание
Проект предназначен для загрузки, обработки и оркестрации данных с использованием Apache Airflow.
Разработка и эксплуатация ведётся на удалённой Ubuntu VM.

## Используемые технологии
- Python
- Apache Airflow
- Git / GitHub
- VS Code Remote SSH
- Ubuntu

## Как работать с проектом
Разработка ведётся напрямую на ВМ через VS Code Remote SSH.  
Изменения версионируются в Git и автоматически подхватываются Airflow.

## Структура проекта
project/
├── dags/
│   └── arima_dag.py            # Для запуска в AirFlow
├── data/
│      ├── model.hd5            # Модель ARIMA
│      ├── predict_arima.csv    # Кол-во логов за час
│      └── recent_logs.csv
│
├── src/
│   ├── __init__.py             # Инициализация пакетов
│   ├── evaluation.py           # И
│   ├── extract.py              # Подготовка данных
│   ├── forecast.py             # Подготовка данных
│   ├── logger_config.py        # Настройка логгирования
│   ├── run_spark.py            # Обучение в Spark
│   ├── transform.py            # Предобработка данных
│   └── utils.py                # Другие функции
│   
├── venv_linux   
├── venv_windows                   
├── .gitignore
├── .stignore
├── Dockerfile
├── logging.log
├── README.md
├── requirements.txt
└── start.py
