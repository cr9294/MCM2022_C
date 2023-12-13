import pandas as pd

# Read the first Excel file
file1 = '回归预测比特币.xls'
df1 = pd.read_excel(file1)

# Read the second Excel file
file2 = '回归预测黄金.xls'
df2 = pd.read_excel(file2)

# Merge the dataframes on the '日期' column
merged_df = pd.merge(df1, df2, on='日期')

# Display the merged dataframe
print(merged_df)

# Save the merged dataframe to a CSV file
merged_df.to_csv('merged_data.csv', index=False)

