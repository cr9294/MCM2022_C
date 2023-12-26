import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('比特币不变变黄金.csv')
#df = pd.read_csv('黄金不变变比特币.csv')
# Print data types to check the type of '日期(月/日/年)' column
print(df.dtypes)

# Convert the '日期' column to datetime format
df['日期(月/日/年)'] = pd.to_datetime(df['日期(月/日/年)'], format='%m/%d/%y')

# Print the filtered DataFrame to check if filtering is working correctly
print(df)

# Filter data for the years from September 2016 to September 2021
start_date = pd.to_datetime('9/12/16', format='%m/%d/%y')
end_date = pd.to_datetime('9/9/21', format='%m/%d/%y')
df_filtered = df[(df['日期(月/日/年)'] >= start_date) & (df['日期(月/日/年)'] <= end_date)]

# Group by year and calculate the total assets for each year
df_filtered['Year'] = df_filtered['日期(月/日/年)'].dt.year
yearly_total_assets = df_filtered.groupby('Year')['资金'].sum()
# Create data for Bitcoin and Gold ratios (adjust data according to your preferences)
bitcoin_ratios = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
gold_ratios = [0, 0.02, 0.04, 0.06, 0.08, 0.1]

#bitcoin_ratios = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
#gold_ratios = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]


'''
# Plot the yearly total assets using a line plot
plt.figure(figsize=(10, 6))
yearly_total_assets.plot(marker='o', linestyle='-')
plt.title('Yearly Total Assets (Sep 2016 - Sep 2021)')
plt.xlabel('Year')
plt.ylabel('Total Assets')
plt.grid(True)
plt.show()

# Create a bar chart for Bitcoin and Gold ratios
years = df_filtered['Year'].unique()
bar_width = 0.35
index = range(len(years))

plt.figure(figsize=(10, 6))
bar1 = plt.bar(index, bitcoin_ratios, bar_width, label='Bitcoin')
bar2 = plt.bar([i + bar_width for i in index], gold_ratios, bar_width, label='Gold')

plt.xlabel('Year')
plt.ylabel('Ratio')
plt.title('Bitcoin and Gold Ratios Over the Years (Sep 2016 - Sep 2021)')
plt.xticks([i + bar_width/2 for i in index], years)
plt.legend()
plt.show()
'''

''''''
years = df_filtered['Year'].unique()
years = [str(year-2016) for year in years]




# Assuming you already have 'yearly_total_assets', 'bitcoin_ratios', 'gold_ratios', and 'years' defined
from matplotlib.ticker import ScalarFormatter
# Create a figure with two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
print(yearly_total_assets)
# Plot Yearly Total Assets in the first subplot
ax1.plot(yearly_total_assets, marker='o', linestyle='-')
ax1.set_title('Yearly Total Assets (Sep 2016 - Sep 2021)')
#ax1.set_xlabel('Year')
ax1.set_ylabel('Total Assets')
ax1.set_xticks(yearly_total_assets.index)
ax1.set_xticklabels(years)
ax1.grid(True)
# 设置y轴不自动缩放，并禁用科学计数法
ax1.get_yaxis().get_major_formatter().set_scientific(False)


# Create a bar chart for Bitcoin and Gold ratios in the second subplot
bar_width = 0.35
index = range(len(years))

bar1 = ax2.bar(index, bitcoin_ratios, bar_width, label='Bitcoin')
bar2 = ax2.bar([i + bar_width for i in index], gold_ratios, bar_width, label='Gold')

#ax2.set_xlabel('Year')
ax2.set_ylabel('Ratio')
ax2.set_title('Bitcoin and Gold Ratios Over the Years (Sep 2016 - Sep 2021)')
ax2.set_xticks([i + bar_width/2 for i in index])
ax2.set_xticklabels(years)
ax2.legend()

# Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.autoscale(axis='y', tight=False)
# Show the combined figure
plt.show()

