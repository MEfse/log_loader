from src.extract import LogDataLoader, LoaderCsvFile, LoadParams, Run, GetInterval

loader = Run()
loader.execute()

#loader = LoaderCsvFile()
#path = LoadParams()
#data = loader.load_recent_logs(path.PATH_LOGS)
#query = GetQuery()
#last_time = query.get_interval(data)

#print(last_time)

#loader = LoadParams()
#print(loader.get_db_params())