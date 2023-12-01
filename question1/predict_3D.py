import pandas as pd
from ultralytics import YOLO
import multiprocessing
from PIL import Image
import os
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    # 加载模型
    model = YOLO('E:\code\python\APMCM\\1(1)\\train5\weights\\best.onnx')  # 加载官方模型

    # 图片文件夹路径
    image_folder = './pic/q1/'

    # 检测结果保存文件夹路径
    output_folder = './runs/predict/'
    os.makedirs(output_folder, exist_ok=True)

    # 获取图片文件列表
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 初始化数据列表
    all_data = {'Image Name': [], 'Box Number': [], 'X Coordinate': [], 'Y Coordinate': []}

    # 循环处理每张图片
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        image_name = os.path.splitext(image_file)[0]
        # 使用模型进行预测
        results = model.predict(image_path, save=False)

        # 提取苹果位置
        apple_positions = [(box[0], box[1]) for box in results[0].boxes.xywh]

        # 添加数据到列表
        all_data['Image Name'].extend([image_name] * len(apple_positions))
        all_data['Box Number'].extend(list(range(1, len(apple_positions) + 1)))
        all_data['X Coordinate'].extend([box[0] for box in apple_positions])
        all_data['Y Coordinate'].extend([box[1] for box in apple_positions])

    # 保存相关信息至CSV文件
    csv_file_path = os.path.join(output_folder, 'apple_positions.csv')
    df = pd.DataFrame(all_data)
    df.to_csv(csv_file_path, index=False)



if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
