import numpy as np
import pandas as pd
import time
from statsmodels.tsa.arima.model import ARIMA
import xlwt

# Load your data
Data = pd.read_csv("../C题处理后的中间文件2.csv")

# Convert the date to a timestamp
def to_timestamp(date):
    return int(time.mktime(time.strptime(date, "%m/%d/%y")))

start_timestamp = to_timestamp(Data.iloc[0, 0])
for i in range(Data.shape[0]):
    Data.iloc[i, 0] = (to_timestamp(Data.iloc[i, 0]) - start_timestamp) / 86400

# Set the number of days for training
days_fit = Data.shape[0]  # Minimum value is 2

# Initialize ARIMA model for Bitcoin
bitcoin_model = ARIMA(Data.iloc[0:days_fit, 1], order=(1, 1, 1))

# Fit the ARIMA model
bitcoin_fit = bitcoin_model.fit()

# Forecasting
bitcoin_forecast = bitcoin_fit.forecast(steps=days_fit)

# Create a DataFrame with the results
result_df = pd.DataFrame({
    '日期': pd.date_range(start="2022-01-01", periods=len(bitcoin_forecast)),  # Replace with your actual start date
    '比特币实际值': Data.iloc[:, 1].values,
    '比特币预测值': bitcoin_forecast.values,
})

# Save the results to Excel
result_df.to_excel("比特币时间序列预测结果.xlsx", index=False)
