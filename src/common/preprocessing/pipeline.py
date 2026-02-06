from dataclasses import dataclass

#Реальная логика: чистка, токенизация, фичи, нормализация.

#Должен быть детерминированным и максимально одинаковым в train/inference.

@dataclass(frozen=True)
class PreprocessResult:
    features: list[float]

def preprocess(text: str) -> PreprocessResult:
    # TODO: заменить на реальную логику
    return PreprocessResult(features=[float(len(text))])


class GetInterval:
    def __init__(self):
        self.params = LoadParams()                          
        self.loader = LoaderCsvFile()

    def get_last_time(self, data: pd.DataFrame) -> Optional[pd.Timestamp]:
        '''Загружает последннюю дату из файла recent_logs

        Args: 
            path_logs (str): Передаем путь для подключения к файлу recent_logs
        Returns:
            last_time (datetime): Возвращаем последнее время в файле recent_logs
        '''
        if data is None or data.empty:
            logger.error("Данных нет.")
            raise
        
        if 'hour' not in data.columns:
            logger.error("Колонка 'hour' отсутствует.")
            raise


        last_time = data['hour'].max()

        if pd.isna(last_time):
            logger.error("Не удалось извлечь дату.")
            raise

        logger.info(f"Получаем последнее время: {last_time}")
        return last_time
    
    def get_interval(self, data: pd.DataFrame) -> Optional[str]:
        '''
        Получение интервала времени для запроса SQL 

        Args: 
            data (pd.DataFrame): Данные по которым считается интервал времени

        Returns:
            interval (str): Интервал недостающих данных
        '''
        last_time = self.get_last_time(data)

        try:
            time_diff = datetime.now() - last_time
            interval = f"{max(1, int(time_diff.total_seconds() / 60))} minutes"
            logger.info(f"Интервал загрузки: {interval}")
            return interval
        except Exception as e:
            logger.error(f"Ошибка при расчёте интервала: {e}")
            raise           