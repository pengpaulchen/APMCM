import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast
import re

# 读取CSV文件
data = pd.read_csv('./runs/predict/count.csv',encoding='gbk')

import pandas as pd
import matplotlib.pyplot as plt


# 提取数据
images = data['图片序号']
apples = data['苹果个数']

# 绘制直方图
plt.bar(images, apples, alpha=0.5, color='gray', edgecolor='black')
plt.xlabel('picture id')
plt.ylabel('apple num')
plt.title('count apple num')
plt.show()
