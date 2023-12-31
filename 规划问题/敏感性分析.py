import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_paths = [
    "资产+每天的实际操作-500-new.csv",
    "资产+每天的实际操作-1000-new.csv",
    "资产+每天的实际操作-2000-new.csv"
]

# 为每个文件指定不同的颜色和标记
line_styles = [
    {'marker': 'o', 'linestyle': '-', 'color': 'b', 'label': '-500'},
    {'marker': 's', 'linestyle': '--', 'color': 'r', 'label': '-1000'},
    {'marker': '^', 'linestyle': ':', 'color': 'g', 'label': '-2000'}
]

# 创建一个图表
plt.figure(figsize=(10, 8))

# 为每个文件绘制线条
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    df['日期'] = pd.to_datetime(df['日期(月/日/年)'], format='%m/%d/%y')
    daily_assets = df.groupby('日期')['资金'].sum()

    # 使用指定的颜色和标记绘制线条
    plt.plot(daily_assets.index, daily_assets.values, **line_styles[i])

# 添加标题和标签
plt.title('Total assets per day')
plt.xlabel('date')
plt.ylabel('Total assets')

# 添加图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()
