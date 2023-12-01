import pandas as pd
import matplotlib.pyplot as plt


# 读取 CSV 文件
csv_file_path = './count_fruits.csv'
df = pd.read_csv(csv_file_path,encoding='gbk')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei'] # 加入一行，解决中文不显示问题，对应字号选择如下图
plt.rcParams['axes.unicode_minus']=False # 解决负号不显示问题
# 绘制分布直方图
plt.bar(df['图片ID号'], df['苹果个数'])
plt.xlabel('Fruit ID')
plt.ylabel('Apple Count')
plt.title('Histogram of the distribution of Fruit ID numbers')
plt.show()
