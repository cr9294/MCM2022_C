import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import csv

BCHAIN_MKPRU=pd.read_csv("BCHAIN-MKPRU.csv",dtype={"Date":str,"Value":np.float64})
LBMA_GOLD=pd.read_csv("LBMA-GOLD.csv",dtype={"Date":str,"Value":np.float64})
Data=pd.read_csv("合并文件.csv")
Data_2=pd.read_csv("合并文件.csv")

def to_timestamp(date):
    return int(time.mktime(time.strptime(date,"%m/%d/%y")))

#将日期变为自然数
start_timestamp=to_timestamp(Data.iloc[0,0])
for i in range(Data.shape[0]):
    Data.iloc[i,0]=(to_timestamp(Data.iloc[i,0])-start_timestamp)/86400


batch_size=1 # 应该只能为1
start_input=50
input_size=Data.shape[0]#训练：通过前input_size天预测input_size+1天，预测：通过2到input_size+1天预测第input_size+2天
hidden_size=20
output_size=1
layers_size=3
lr=10
num_epochs=1000


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers_size):
        super().__init__()
        self.GRU_layer = nn.GRU(input_size, hidden_size, layers_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.GRU_layer(x)
        x = self.linear(x)
        return x

device=torch.device("cuda")

gru=GRUModel(30, hidden_size, output_size, layers_size).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(gru.parameters(), lr)

ji=np.array(Data.iloc[0:input_size,1].dropna())
input_size=ji.shape[0]-2

trainB_x=torch.from_numpy(ji[input_size-30:input_size].reshape(-1,batch_size,30)).to(torch.float32).to(device)
trainB_y=torch.from_numpy(ji[input_size].reshape(-1,batch_size,output_size)).to(torch.float32).to(device)

losses = []

for epoch in range(num_epochs):
    output = gru(trainB_x).to(device)
    loss = criterion(output, trainB_y)
    losses.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss" + str(epoch) + ":", loss.item())


pred_x_train=torch.from_numpy(ji[input_size-29:input_size+1]).reshape(-1,1,30).to(torch.float32).to(device)
pred_y_train=gru(pred_x_train).to(device)
print("prediction:",pred_y_train.item())
print("actual:",ji[input_size+1])


# 预测代码
losses = []
predictions = []
actuals = []
for i in range(start_input, input_size + 1, 1):#修改input_size，可以改变预测的天数
    print("进行到input_size=", i)
    # gru=GRUModel(i, hidden_size, output_size, layers_size).to(device)
    gru = GRUModel(30, hidden_size, output_size, layers_size).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(gru.parameters(), lr)

    # 数据，以比特币为例
    trainB_x = torch.from_numpy(ji[i - 30:i].reshape(-1, batch_size, 30)).to(torch.float32).to(device)
    trainB_y = torch.from_numpy(ji[i].reshape(-1, batch_size, output_size)).to(torch.float32).to(device)

    loss = None

    for epoch in range(num_epochs):
        output = gru(trainB_x).to(device)
        loss = criterion(output, trainB_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("loss"+str(epoch)+":", loss.item())
    losses.append(loss.item())

    # 预测，以比特币为例
    pred_x_train = torch.from_numpy(ji[i - 29:i + 1].reshape(-1, 1, 30)).to(torch.float32).to(device)
    pred_y_train = gru(pred_x_train).to(device)
    # print("prediction:",pred_y_train.item())
    # print("actual:",Data.iloc[i+1,1])
    predictions.append(pred_y_train.item())
    actuals.append(ji[i + 1])

plt.figure(figsize=(20, 6))  # 调整图形大小
plt.plot(losses, label='Loss', linestyle='-', color='yellow')
plt.plot(predictions, label='Prediction', linestyle='--', color='purple')
plt.plot(actuals, label='Real', linestyle='-.', color='orange')
plt.legend()
plt.title('Comparison of Loss, Prediction, and Real Values')
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.grid(True)
plt.xticks(rotation=45)
plt.xticks(range(0, len(losses),100))  # 调整 X 轴刻度间隔
plt.show()



print(np.array(predictions).shape[0])
print(np.array(actuals).shape[0])
print(np.array(losses).shape[0])
print(input_size-29)

# 保存预测结果
with open('预测结果比特币.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['日期','预测值', '实际值', '损失'])
    for i in range(np.array(predictions).shape[0]):
        writer.writerow([Data_2.iloc[i+51,0],predictions[i], actuals[i], losses[i]])



