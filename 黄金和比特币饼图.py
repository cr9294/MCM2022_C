import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
Data = pd.read_csv("合并文件.csv")

# Extract the third column values
column3_values = Data.iloc[:, 2]

# Create a pie chart
labels = ['0', '1']
colors = ['#ff9999', '#66b3ff']
explode = (0.1, 0)  # Explode the first slice (value 0) for emphasis

# Calculate the count of values 0 and 1
count_0 = sum(column3_values == 0)
count_1 = sum(column3_values == 1)
values = [count_0, count_1]

# Calculate the total count
total = len(column3_values)

# Plot the pie chart
plt.pie(values, labels=[f'0: {count_0}', f'1: {count_1}'], autopct='%1.1f%%', startangle=90, colors=colors, explode=explode)
plt.title(f'Distribution of 0 and 1 (Total: {total})')
plt.show()
