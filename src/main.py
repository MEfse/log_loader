from src.loader import load_data_from_db
from src import preprocess_arima


def main():
    data = load_data_from_db()
    if not data.empty:
        ts = preprocess_arima(data)


if __name__ == "__main__":
    main()


# python -m src.main
