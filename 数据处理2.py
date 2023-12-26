import pandas as pd

# 读取CSV文件
file_path = '合并文件.csv'  # 请替换成你的文件路径
df = pd.read_csv(file_path)

# 将空白的黄金价值替换为null
df['黄金价值'].fillna('null', inplace=True)

# 保存修改后的DataFrame到CSV文件
df.to_csv('合并文件.csv', index=False)

print("处理完成")
