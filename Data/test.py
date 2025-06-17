import pandas as pd

data = pd.read_csv('Data/Housing.csv')

print(data.head())

print(data.isna().sum())