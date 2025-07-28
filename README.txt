project/
├── data/
│   ├── ARIMA/
│   │   ├── predict_arima.csv
│   │   └── recent_logs.csv
│   ├── Others_Model/
│   │   ├── internal_connection.csv
│   │   └── external_connection.csv
│
├── models/
│   └── model.hd5
│
├── src/
│   ├── __init__.py
│   ├── DAG.py
│   ├── evaluation.py

│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── run_spark.py
│   ├── utils.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── logger_config.py
│   │   └── path_config.py
│   │
│   ├── loader/
│       ├── __init__.py
│       ├── connect_to_db.py 
│       ├── loader_csv_file.py   
│       └── loader_model.py
│
├── venv/                     # виртуальное окружение (в .gitignore)
├── .gitignore
├── logging.log
├── README.md
├── requirements.txt
├── start.py
└── test.py