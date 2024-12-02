import pandas as pd

df = pd.read_parquet('test-00000-of-00001.parquet')

df.to_csv('test-00000-of-00001.csv', index=False)