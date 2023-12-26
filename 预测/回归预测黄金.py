import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
import xlwt

BCHAIN_MKPRU = pd.read_csv("../BCHAIN-MKPRU.csv", dtype={"Date": str, "Value": np.float64})
LBMA_GOLD = pd.read_csv("../LBMA-GOLD.csv", dtype={"Date": str, "Value": np.float64})
Data = pd.read_csv("../合并文件.csv")
df = pd.read_csv("../合并文件.csv")

def to_timestamp(date):
    return int(time.mktime(time.strptime(date, "%m/%d/%y")))

# 将日期变为自然数
start_timestamp = to_timestamp(Data.iloc[0, 0])
for i in range(Data.shape[0]):
    Data.iloc[i, 0] = (to_timestamp(Data.iloc[i, 0]) - start_timestamp) / 86400

days_fit = Data.shape[0]  # 最小为2

bFit = Data.iloc[0:days_fit, 0:2]
gFit = Data.iloc[0:days_fit, 0::3].dropna()  # 需要考虑NaN的问题

bitcoin_reg = linear_model.LinearRegression()
gold_reg = linear_model.LinearRegression()

bitcoin_reg.fit(np.array(bFit.iloc[:, 0]).reshape(-1, 1), np.array(bFit.iloc[:, 1]).reshape(-1, 1))
gold_reg.fit(np.array(gFit.iloc[:, 0]).reshape(-1, 1), np.array(gFit.iloc[:, 1]).reshape(-1, 1))

# 预测代码：
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

ji1 = np.array(g_pred_linear).reshape(-1, 1)
ji1 = np.array(ji1)
ji2 = Data.iloc[2:days_fit + 1, 3]
ji2 = np.array(ji2)

import csv

# Assuming df, ji1, and ji2 are defined earlier in your code

with open("回归预测黄金.csv", mode="w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)

    col = ["日期", "标签", "真实值", "预测值", "误差"]  # List
    csv_writer.writerow(col)
    # Lists to store predicted values for plotting
    predicted_values = []
    real_values = []

    csv_writer.writerow(["9/12/16", 0, 1324.6, 1324.6, 0])  # Data for row 1

    for i in range(0, 1824):
        print("第%d条" % (i + 1))
        if df.values[i + 2][2] != 1:
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
# Plotting predicted vs real values
plt.plot(real_values, label='Real Values', marker='o', color='blue', linestyle='-')
plt.plot(predicted_values, label='Predicted Values', marker='x', color='yellow', linestyle='--')

plt.xlabel('Data Points')
plt.ylabel('Values')
plt.title('Predicted vs Real Values(Gold)')
plt.legend()
plt.show()