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
Data = pd.read_csv("../C题处理后的中间文件2.csv")
df = pd.read_csv("../C题处理后的中间文件2.csv")

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
ji2 = np.array(Data.iloc[2:days_fit + 1, 1])
ji3 = [round(ji1[i][0][0][0], 2) for i in range(2, 1826)]

book = xlwt.Workbook(encoding="utf-8", style_compression=0)
sheet = book.add_sheet("回归预测比特币", cell_overwrite_ok=True)
col = ("日期", "预测值", "真实值", "误差")

for i in range(0, 4):
    sheet.write(0, i, col[i])

for i in range(0, 1824):
    sheet.write(i + 3, 0, Data.values[i + 2][0])
    sheet.write(i + 3, 1, ji3[i])
    sheet.write(i + 3, 2, ji2[i])
    sheet.write(i + 3, 3, abs(ji3[i] - ji2[i]))

book.save("回归预测比特币.xls")
