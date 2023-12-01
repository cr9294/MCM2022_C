import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
from d2l import torch as d2l
import torch
import torch.nn as nn
import csv

# 设置设备为 CPU
device = torch.device("cpu")

# 读取数据
BCHAIN_MKPRU = pd.read_csv("../BCHAIN-MKPRU.csv", dtype={"Date": np.str, "Value": np.float64})
LBMA_GOLD = pd.read_csv("../LBMA-GOLD.csv", dtype={"Date": np.str, "Value": np.float64})
Data = pd.read_csv("../C题处理后的中间文件2.csv")

# 将日期转换为时间戳
def to_timestamp(date):
    return int(time.mktime(time.strptime(date, "%m/%d/%y")))

# 将日期变为自然数
start_timestamp = to_timestamp(Data.iloc[0, 0])
for i in range(Data.shape[0]):
    Data.iloc[i, 0] = (to_timestamp(Data.iloc[i, 0]) - start_timestamp) / 86400

# 打印处理后的数据
print(Data)

# 模型参数
start_input = 1000
batch_size = 1
input_size = Data.shape[0]
hidden_size = 20
output_size = 1
layers_size = 3
lr = 0.01  # 修改学习率
num_epochs = 1000

# 定义 GRU 模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers_size):
        super().__init__()
        self.GRU_layer = nn.GRU(input_size, hidden_size, layers_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.GRU_layer(x)
        x = self.linear(x)
        return x

# 创建 GRU 模型实例
gru = GRUModel(30, hidden_size, output_size, layers_size).to(device)

# 损失函数和优化器
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(gru.parameters(), lr=lr)

# 提取比特币价格数据
bitcoin_prices = np.array(Data.iloc[0:input_size, 3].dropna())
input_size = bitcoin_prices.shape[0] - 2

# 准备训练数据
trainB_x = torch.from_numpy(bitcoin_prices[input_size - 30:input_size].reshape(-1, batch_size, 30)).to(torch.float32).to(device)
trainB_y = torch.from_numpy(bitcoin_prices[input_size].reshape(-1, batch_size, output_size)).to(torch.float32).to(device)

# 训练模型
losses = []
for epoch in range(num_epochs):
    output = gru(trainB_x).to(device)
    loss = criterion(output, trainB_y)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("损失" + str(epoch) + ":", loss.item())

# 预测比特币价格
pred_x_train = torch.from_numpy(bitcoin_prices[input_size - 29:input_size + 1]).reshape(-1, 1, 30).to(torch.float32).to(device)
pred_y_train = gru(pred_x_train).to(device)
print("预测值:", pred_y_train.item())
print("实际值:", bitcoin_prices[input_size + 1])

# 绘制损失曲线
plt.plot(losses)
plt.show()

# 预测代码
losses = []
predictions = []
actuals = []
for i in range(start_input, input_size + 1):
    print("进行到input_size=", i)
    # 创建新的模型实例
    gru = GRUModel(30, hidden_size, output_size, layers_size).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(gru.parameters(), lr=lr)

    # 准备训练数据
    trainB_x = torch.from_numpy(bitcoin_prices[i - 30:i].reshape(-1, batch_size, 30)).to(torch.float32).to(device)
    trainB_y = torch.from_numpy(bitcoin_prices[i].reshape(-1, batch_size, output_size)).to(torch.float32).to(device)

    # 训练模型
    loss = None
    for epoch in range(num_epochs):
        output = gru(trainB_x).to(device)
        loss = criterion(output, trainB_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("损失"+str(epoch)+":", loss.item())
    losses.append(loss.item())

    # 预测
    pred_x_train = torch.from_numpy(bitcoin_prices[i - 29:i + 1].reshape(-1, 1, 30)).to(torch.float32).to(device)
    pred_y_train = gru(pred_x_train).to(device)
    predictions.append(pred_y_train.item())
    actuals.append(bitcoin_prices[i + 1])

# 绘制预测结果和实际值
plt.plot(predictions, label='real')
plt.plot(actuals, label='pred')
plt.legend()
plt.show()

# 写入预测结果到 CSV 文件
f = open('周期lstm黄金预测.csv', 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["实际价格", "预测价格"])

# Use min length to avoid IndexErrors
min_length = min(len(actuals), len(predictions))
for i in range(min_length):
    tmp = [actuals[i], round(predictions[i], 2)]
    csv_writer.writerow(tmp)

f.close()

