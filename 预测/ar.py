import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from statsmodels.tsa.arima.model import ARIMA
import xlwt
import openpyxl

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
bitcoin_model = ARIMA(Data.iloc[0:days_fit, 1], order=(5, 1, 0))  # Replace p, d, q with appropriate values

# Fit the ARIMA model
bitcoin_fit = bitcoin_model.fit()

# Forecasting
bitcoin_forecast = bitcoin_fit.forecast(steps=days_fit)  # Adjust steps as needed

# Prepare the results
ji3 = np.round(bitcoin_forecast, 2)

# Reset the index before extracting values from column 1
ji2 = np.array(Data.iloc[2:days_fit + 1, 1].reset_index(drop=True))
# 在提取第1列的值之前重置索引
# 截取 ji3，使其长度与 ji2 相同
ji3 = np.round(bitcoin_forecast[:len(ji2)], 2)
print(ji3)
print(ji2)
# 合并 ji2 和 ji3 成一个 DataFrame
result_df = pd.DataFrame({'日期': Data.iloc[2:days_fit + 1, 0].values,
                          '预测值': ji3,
                          '真实值': ji2,
                          '误差': np.abs(ji3 - ji2)})

# 保存结果到 Excel 文件
result_df.to_excel("ARIMA预测比特币结果.xlsx", index=False)
# Save the results to Excel
book = xlwt.Workbook(encoding="utf-8", style_compression=0)
sheet = book.add_sheet("ARIMA预测比特币", cell_overwrite_ok=True)
col = ("日期", "预测值", "真实值", "误差")

for i in range(0, 4):
    sheet.write(0, i, col[i])

# Fill in the sheet
for i in range(0, days_fit - 1):
    sheet.write(i + 1, 0, Data.iloc[i + 2, 0])
    sheet.write(i + 1, 1, ji3[i][0])
    sheet.write(i + 1, 2, ji2[i])
    sheet.write(i + 1, 3, abs(ji3[i][0] - ji2[i]))

# Save the workbook
book.save("ARIMA预测比特币.xls")
