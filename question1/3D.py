import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast
import re

# 读取CSV文件
df = pd.read_csv('./runs/predict/apple_positions.csv')

# 自定义函数，用于提取张量值
def extract_tensor_value(tensor_str):
    match = re.search(r'tensor\((.*?)\)', tensor_str)
    if match:
        return float(match.group(1))
    else:
        return None

# 应用自定义函数以提取张量值
df['X Coordinate'] = df['X Coordinate'].apply(extract_tensor_value)
df['Y Coordinate'] = df['Y Coordinate'].apply(extract_tensor_value)

# 创建3D散点图
# 创建3D散点图，调整图形大小为10x8
fig = plt.figure(figsize=(20, 16))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
for index, row in df.iterrows():
    x = int(row['Image Name'])
    y = row['X Coordinate']
    z = row['Y Coordinate']
    ax.scatter(x, y, z)

# 设置坐标轴标签
ax.set_ylabel('Image Name')
ax.set_xlabel('X Coordinate')
ax.set_zlabel('Y Coordinate')

# 显示图形
plt.show()
