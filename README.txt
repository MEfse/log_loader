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
