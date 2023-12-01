# 导入所需的库
import pandas as pd
import csv
import numpy as np

# 读取CSV文件到DataFrame
df = pd.read_csv("../C题处理后的中间文件2.csv")

# 获取DataFrame的行数和列数
height, weight = df.shape

# 将DataFrame转换为NumPy数组
m = df.values

# 从数组中提取黄金数据
huangjin = []
for i in range(0, height):
    if (m[i][2] == 0):
        huangjin.append(m[i][3])

# 初始化变量
k = 1   # k代表现在持有的是黄金还是比特币
n = 0   # 黄金统计

rateG = 0.01  # 黄金的交易费率
rateB = 0.02  # 比特币的交易费率

oldprice_G = 0
oldprice_B = 0
newprice_G = 0
newprice_B = 0
target = 1000  # 初始资产

res = []  # 存储每天的操作

# 循环遍历每一天的数据
for i in range(1, height - 1):
    res1 = []

    # 打印当前天的资产
    print("第%d天的资产为：%f" % (i, target))

    # 如果黄金不能交易
    if (m[i][2] == 1):
        # 持有现金
        if (k == 0):
            oldprice_B = m[i][1]
            newprice_B = m[i + 1][1]
            tmp = target / (1 + rateB) * newprice_B / oldprice_B
            tmp = round(tmp, 2)
            target1 = max(target, tmp)
            if (target1 == target):
                k1 = 0
            else:
                k1 = 1

        # 持有比特币
        if (k == 1):
            oldprice_B = m[i][1]
            newprice_B = m[i + 1][1]
            tmp1 = target * (newprice_B / oldprice_B)
            tmp1 = round(tmp1, 2)
            tmp2 = target / (1 + rateB)
            tmp2 = round(tmp2, 2)
            target1 = max(tmp1, tmp2)
            if (target1 == tmp2):
                k1 = 0
            else:
                k1 = 1

        # 持有黄金
        if (k == 2):
            target1 = target
            k1 = 2

        res1.append(m[i][0])
        res1.append(k1)
        res.append(res1)
        k = k1
        target = target1

    # 如果黄金可交易
    if (m[i][2] == 0):
        oldprice_B = m[i][1]
        newprice_B = m[i + 1][1]
        oldprice_G = huangjin[n]
        newprice_G = huangjin[n + 1]

        # 持有现金
        if (k == 0):
            tmp1 = target / (1 + rateG) * newprice_G / oldprice_G
            tmp1 = round(tmp1, 2)
            tmp2 = target / (1 + rateB) * newprice_B / oldprice_B
            tmp2 = round(tmp2, 2)
            target1 = max(tmp1, tmp2, target)
            if (target1 == target):
                k1 = 0
            if (target1 == tmp1):
                k1 = 2
            if (target1 == tmp2):
                k1 = 1

        # 持有比特币
        if (k == 1):
            tmp1 = target * newprice_B / oldprice_B
            tmp1 = round(tmp1, 2)
            tmp2 = target / (1 + rateB)
            tmp2 = round(tmp2, 2)
            tmp3 = target / (1 + rateB + rateG) * newprice_G / oldprice_G
            tmp3 = round(tmp3, 3)
            target1 = max(tmp1, tmp2, tmp3)
            if (target1 == tmp1):
                k1 = 1
            if (target1 == tmp2):
                k1 = 0
            if (target1 == tmp3):
                k1 = 2

        # 持有黄金
        if (k == 2):
            tmp1 = target * newprice_G / oldprice_G
            tmp1 = round(tmp1, 2)
            tmp2 = target / (1 + rateG)
            tmp2 = round(tmp2, 2)
            tmp3 = target / (1 + rateB + rateG) * newprice_B / oldprice_B
            tmp3 = round(tmp3, 3)
            target1 = max(tmp1, tmp2, tmp3)
            if (target1 == tmp1):
                k1 = 2
            if (target1 == tmp2):
                k1 = 0
            if (target1 == tmp3):
                k1 = 1

        n = n + 1
        res1.append(m[i][0])
        res1.append(k1)
        res.append(res1)
        k = k1
        target = target1

# 将结果转换为NumPy数组
res = np.array(res)
print(res.shape[0])

# 打印最后一天的资产
print("第%d天的资产为：%f" % (height - 1, target))

# 将结果写入CSV文件
f = open('每天的实际操作.csv', 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["日期(月/日/年)", "操作"])
csv_writer.writerow(["9/11/16", 0])
for i in range(0, 1824):
    csv_writer.writerow(res[i])
f.close()
