import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
# from d2l import torch as d2l  # 如果此行代码未被使用，建议注释掉
import torch
import torch.nn as nn
import xlwt

# 从CSV文件中读取数据
BCHAIN_MKPRU = pd.read_csv("../BCHAIN-MKPRU.csv", dtype={"Date": str, "Value": np.float64})
LBMA_GOLD = pd.read_csv("../LBMA-GOLD.csv", dtype={"Date": str, "Value": np.float64})
Data = pd.read_csv("../合并文件.csv")
df = pd.read_csv("../合并文件.csv")

# 将日期转换为时间戳的函数
def to_timestamp(date):
    return int(time.mktime(time.strptime(date, "%m/%d/%y")))

# 将日期转换为自然数
start_timestamp = to_timestamp(Data.iloc[0, 0])
for i in range(Data.shape[0]):
    Data.iloc[i, 0] = (to_timestamp(Data.iloc[i, 0]) - start_timestamp) / 86400

days_fit = Data.shape[0]  # 最小为2

bFit = Data.iloc[0:days_fit, 0:2]
gFit = Data.iloc[0:days_fit, 0::3].dropna()  # 需要考虑NaN的问题
print(bFit)
print(gFit)
bitcoin_reg = linear_model.LinearRegression()
gold_reg = linear_model.LinearRegression()

bitcoin_reg.fit(np.array(bFit.iloc[:, 0]).reshape(-1, 1), np.array(bFit.iloc[:, 1]).reshape(-1, 1))
gold_reg.fit(np.array(gFit.iloc[:, 0]).reshape(-1, 1), np.array(gFit.iloc[:, 1]).reshape(-1, 1))

# print("bitcoin:",bitcoin_reg.predict(np.array([days_fit]).reshape(-1,1)))
# print("gold:",gold_reg.predict(np.array([days_fit]).reshape(-1,1)))

b_pred_linear = [None, None]
g_pred_linear = [None, None]

for day_fit in range(2, days_fit + 1):
    bFit = Data.iloc[0:day_fit, 0:2]
    gFit = Data.iloc[0:day_fit, 0::3].dropna()

    bitcoin_reg = linear_model.LinearRegression()
    gold_reg = linear_model.LinearRegression()

    bitcoin_reg.fit(np.array(bFit.iloc[:, 0]).reshape(-1, 1), np.array(bFit.iloc[:, 1]).reshape(-1, 1))
    gold_reg.fit(np.array(gFit.iloc[:, 0]).reshape(-1, 1), np.array(gFit.iloc[:, 1]).reshape(-1, 1))

    b_pred_linear.append(bitcoin_reg.predict(np.array([day_fit]).reshape(-1, 1)))
    g_pred_linear.append(gold_reg.predict(np.array([day_fit]).reshape(-1, 1)))

ji1 = np.array(b_pred_linear).reshape(-1, 1)
ji1 = np.array(ji1)
ji2 = np.array(Data.iloc[2:days_fit + 1, 1])
ji2 = np.array(ji2)
print(ji2)

import csv

# Assuming df, ji1, and ji2 are defined earlier in your code

with open("回归预测比特币.csv", mode="w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)

    col = ["日期", "标签", "预测值", "真实值", "误差"]
    csv_writer.writerow(col)
    # Lists to store predicted values for plotting
    predicted_values = []
    real_values = []

    for i in range(0, 1824):
        print("第%d条" % (i + 1))
        if df.values[i + 2][2] == 1 or df.values[i + 2][2] == 0:
            row_data = [
                df.values[i + 2][0],
                df.values[i + 2][2],
                ji2[i],
                ji1[i + 2][0][0][0],
                abs(ji1[i + 2][0][0][0] - ji2[i])
            ]
            csv_writer.writerow(row_data)

            # Append values for plotting
            predicted_values.append(ji1[i + 2][0][0][0])
            real_values.append(ji2[i])

# Plotting predicted vs real values
plt.plot(real_values, label='Real Values', marker='o', color='green', linestyle='-')
plt.plot(predicted_values, label='Predicted Values', marker='x', color='red', linestyle='--')

plt.xlabel('Data Points')
plt.ylabel('Values')
plt.title('Predicted vs Real Values(Bitcoin)')
plt.legend()
plt.show()