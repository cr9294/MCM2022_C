import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Read the data for daily operations
operations_df = pd.read_csv('资产+每天的实际操作-1000.csv')

# Convert the date column to datetime type
operations_df['日期(月/日/年)'] = pd.to_datetime(operations_df['日期(月/日/年)'], format='%m/%d/%y')

# Extract year from the date
operations_df['Year'] = operations_df['日期(月/日/年)'].dt.year

# Set the plotting style using seaborn
sns.set(style='darkgrid')

# Plot the daily operations
plt.figure(figsize=(12, 6))
plt.plot(operations_df['日期(月/日/年)'], operations_df['操作'], marker='o', linestyle='-', color='#4287f5', label='Actual Operations')
plt.title('Daily Operations', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Operation (0: Cash, 1: Bitcoin, 2: Gold)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks([0, 1, 2], ['Cash', 'Bitcoin', 'Gold'], fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

# Add statistical information to the plot
for operation_type in [0, 1, 2]:
    total_days = operations_df[operations_df['操作'] == operation_type].shape[0]
    plt.text(operations_df['日期(月/日/年)'].iloc[-1], operation_type, f'Total Days: {total_days}', fontsize=10,
             va='center', ha='left', color='black', bbox=dict(facecolor='white', alpha=0.7))

# Show the plot
plt.show()

# Count occurrences of 0, 1, and 2 for each year
occurrences_by_year = operations_df.groupby(['Year', '操作']).size().unstack(fill_value=0)

# Print occurrences
print("\nOccurrences by Year:")
print(occurrences_by_year)

import matplotlib.pyplot as plt

# Your existing code

# Add statistical information to the plot
for operation_type in [0, 1, 2]:
    total_days = operations_df[operations_df['操作'] == operation_type].shape[0]
    plt.text(operations_df['日期(月/日/年)'].iloc[-1], operation_type, f'Total Days: {total_days}', fontsize=10,
             va='center', ha='left', color='black', bbox=dict(facecolor='white', alpha=0.7))

# Show the plot
plt.show()

# Count occurrences of 0, 1, and 2 for each year
occurrences_by_year = operations_df.groupby(['Year', '操作']).size().unstack(fill_value=0)

# Plot occurrences as a bar plot
occurrences_by_year.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Occurrences by Year')
plt.xlabel('Year')
plt.ylabel('Occurrences')

# Add a line plot for total occurrences
total_occurrences_by_year = occurrences_by_year.sum(axis=1)
plt.plot(total_occurrences_by_year.index, total_occurrences_by_year.values, color='black', marker='o', label='Total', linestyle='dashed')

# Show legend
plt.legend()

# Show the combined plot
plt.show()

