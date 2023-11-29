import pandas as pd
import csv
import numpy as np

# Replace '回归预测比特币.xls' with the path to your Excel file
file_path = '回归预测比特币.xls'

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# Convert the DataFrame values to a NumPy array
df_array = np.array(df.values)

# Access the second column (column index 1) in the NumPy array
second_column = df_array[:, 1]

# Print or use the second column as needed
print(second_column)

