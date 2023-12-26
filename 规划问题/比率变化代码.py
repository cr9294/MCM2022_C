import pandas as pd
import csv
import numpy as np


def process_data(file_path):
    df = pd.read_csv(file_path)
    height, _ = df.shape
    m = df.values
    huangjin = [m[i][3] for i in range(height) if m[i][2] == 0]
    return df, height, m, huangjin


def calculate_asset_value_and_operations(df, height, m, huangjin):
    # Initialization
    k = 1  # k represents holding gold (2 for gold, 1 for bitcoin, 0 for cash)
    n = 0  # Gold statistics
    rateG = 0  # Gold transaction fee rate
    rateB = 0.02  # Bitcoin transaction fee rate
    target = 500  # Initial assets
    res = []  # Store daily operations

    # Main Loop
    for i in range(1, height - 1):
        res1 = []

        # Print current day's assets
        #print("Day %d's assets: %f" % (i, target))
        res1.append(target)
        #print(m[i][0])

        if m[i][0] == "1/1/17":
            #rateB += 0.02
            rateG += 0.02
            print("rateB:", rateB)
            print("rateG:", rateG)

        if m[i][0] == "1/1/18":
            #rateB += 0.02
            rateG += 0.02
            print("rateB:", rateB)
            print("rateG:", rateG)

        if m[i][0] == "1/1/19":
            #rateB += 0.02
            rateG += 0.02
            print("rateB:", rateB)
            print("rateG:", rateG)

        if m[i][0] == "1/1/20":
            #rateB += 0.02
            rateG += 0.02
            print("rateB:", rateB)
            print("rateG:", rateG)

        if m[i][0] == "1/1/21":
            #rateB += 0.02
            rateG += 0.02
            print("rateB:", rateB)
            print("rateG:", rateG)

        # If gold cannot be traded
        if m[i][2] == 1:
            # Holding cash
            if k == 0:
                oldprice_B = m[i][1]
                newprice_B = m[i + 1][1]
                tmp = target / (1 + rateB) * newprice_B / oldprice_B
                tmp = round(tmp, 2)
                target1 = max(target, tmp)
                k1 = 0 if target1 == target else 1

            # Holding bitcoin
            elif k == 1:
                oldprice_B = m[i][1]
                newprice_B = m[i + 1][1]
                tmp1 = target * (newprice_B / oldprice_B)
                tmp1 = round(tmp1, 2)
                tmp2 = target / (1 + rateB)
                tmp2 = round(tmp2, 2)
                target1 = max(tmp1, tmp2)
                k1 = 0 if target1 == tmp2 else 1

            # Holding gold
            elif k == 2:
                target1 = target
                k1 = 2

        # If gold can be traded
        elif m[i][2] == 0:
            oldprice_B = m[i][1]
            newprice_B = m[i + 1][1]
            oldprice_G = huangjin[n]
            newprice_G = huangjin[n + 1]

            # Holding cash
            if k == 0:
                tmp1 = target / (1 + rateG) * newprice_G / oldprice_G
                tmp1 = round(tmp1, 2)
                tmp2 = target / (1 + rateB) * newprice_B / oldprice_B
                tmp2 = round(tmp2, 2)
                target1 = max(tmp1, tmp2, target)
                k1 = 2 if target1 == tmp1 else 1 if target1 == tmp2 else 0

            # Holding bitcoin
            elif k == 1:
                tmp1 = target * newprice_B / oldprice_B
                tmp1 = round(tmp1, 2)
                tmp2 = target / (1 + rateB)
                tmp2 = round(tmp2, 2)
                tmp3 = target / (1 + rateB + rateG) * newprice_G / oldprice_G
                tmp3 = round(tmp3, 3)
                target1 = max(tmp1, tmp2, tmp3)
                k1 = 1 if target1 == tmp1 else 0 if target1 == tmp2 else 2

            # Holding gold
            elif k == 2:
                tmp1 = target * newprice_G / oldprice_G
                tmp1 = round(tmp1, 2)
                tmp2 = target / (1 + rateG)
                tmp2 = round(tmp2, 2)
                tmp3 = target / (1 + rateB + rateG) * newprice_B / oldprice_B
                tmp3 = round(tmp3, 3)
                target1 = max(tmp1, tmp2, tmp3)
                k1 = 2 if target1 == tmp1 else 0 if target1 == tmp2 else 1

            n += 1

        res1.extend([m[i][0], k1])
        res.append(res1)
        k = k1
        target = target1

    return np.array(res)


def write_to_csv(results, output_file):
    with open(output_file, 'w', encoding='utf-8', newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["资金", "日期(月/日/年)", "操作"])
        csv_writer.writerows(results)


if __name__ == "__main__":
    file_path = "../合并文件.csv"
    df, height, m, huangjin = process_data(file_path)
    results = calculate_asset_value_and_operations(df, height, m, huangjin)

    # Output
    print("Number of rows in the resulting array:", results.shape[0])
    print(results)

    output_file = '比特币不变变黄金.csv'
    write_to_csv(results, output_file)
