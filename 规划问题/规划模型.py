import pandas as pd
import numpy as np

# Function to read Excel file and return reshaped DataFrame
def read_and_reshape_excel(file_path, column_index):
    df = pd.read_excel(file_path)
    df_array = np.array(df.values)
    df_column = np.array(df_array[:, column_index]).reshape(-1, 1)
    return df_column

# Read Bitcoin data
bitcoin_file_path = '../回归预测比特币.xls'
bitcoin_prices = read_and_reshape_excel(bitcoin_file_path, 1)
print(bitcoin_prices)
# Read Gold data
gold_file_path = '../回归预测黄金.xls'
gold_prices = read_and_reshape_excel(gold_file_path, 3)
print(gold_prices)

# Initialize variables
k = 1  # k represents whether the current holding is gold or bitcoin
target = 1000
res = []

# Iterate through the data
for i in range(1, len(bitcoin_prices) - 1):
    res1 = [target]
    res.append(res1)
    print(f"第{i}天的资产为：{target}")

    if bitcoin_prices[i][0] == 1:  # If gold cannot be traded
        old_price = bitcoin_prices[i - 1][0]
        new_price = bitcoin_prices[i][0]

        if k == 0:  # Holding cash
            target = max(target, round(target / (1 + 0.02) * new_price / old_price, 2))
        elif k == 1:  # Holding bitcoin
            target = max(target * (new_price / old_price), round(target / (1 + 0.02), 2))
        elif k == 2:  # Holding gold
            target = target

    if bitcoin_prices[i][0] == 0:  # If gold can be traded
        old_price_btc = bitcoin_prices[i - 1][0]
        new_price_btc = bitcoin_prices[i][0]
        old_price_gold = gold_prices[i - 1][0]
        new_price_gold = gold_prices[i][0]

        if k == 0:  # Holding cash
            target = max(
                round(target / (1 + 0.01) * new_price_gold / old_price_gold, 2),
                round(target / (1 + 0.02) * new_price_btc / old_price_btc, 2),
                target
            )
        elif k == 1:  # Holding bitcoin
            target = max(
                target * (new_price_btc / old_price_btc),
                round(target / (1 + 0.02), 2),
                round(target / (1 + 0.01 + 0.02) * new_price_gold / old_price_gold, 2)
            )
        elif k == 2:  # Holding gold
            target = max(
                target * (new_price_gold / old_price_gold),
                round(target / (1 + 0.01), 2),
                round(target / (1 + 0.01 + 0.02) * new_price_btc / old_price_btc, 2)
            )

    k = 2 if bitcoin_prices[i][0] == 0 else k

# Final day
res.append([target])

# Convert the result to a NumPy array
res = np.array(res)

# Display the final result
print(f"第{len(bitcoin_prices) - 1}天的资产为：{target}")

# Optionally, write the result to a CSV file
# res_df = pd.DataFrame(res, columns=["资金（少一天）"])
# res_df.to_csv('最终版初始资金为500的资金变化.csv', encoding='utf-8', index=False)
