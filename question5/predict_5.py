from ultralytics import YOLO
import multiprocessing
import os
import csv
import matplotlib.pyplot as plt
import re

def extract_image_id(file_name):
    # 从文件名中提取图片ID号
    match = re.search(r'\((\d+)\)', file_name)
    if match:
        return match.group(1)
    else:
        return None

def main():
    # 加载模型
    model = YOLO('./runs/detect/train4/weights/best.onnx')  # 加载官方模型

    # 图片文件夹路径
    image_folder = './Attachment 3/'

    # 检测结果保存文件夹路径
    output_folder = './runs/predict/'
    os.makedirs(output_folder, exist_ok=True)

    # 获取图片文件列表
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 用于保存结果的 CSV 文件
    csv_file_path = os.path.join(output_folder, 'count_fruits.csv')

    with open(csv_file_path, 'w', newline='') as csvfile:
        # 定义 CSV 写入对象
        csv_writer = csv.writer(csvfile)

        # 写入 CSV 头部
        csv_writer.writerow(['图片ID号', '苹果个数'])

        # 用于存储所有图片的苹果个数和ID号
        apple_count_list = []

        # 循环处理每张图片
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_folder, image_file)
            image_name = os.path.splitext(image_file)[0]

            # 使用模型进行预测
            results = model.predict(image_path, save=False)

            # 获取图片ID号
            image_id = extract_image_id(image_name)

            if image_id is not None:
                # 获取检测到的苹果个数
                apple_count = len(results[0].boxes)

                # 将图片ID号和苹果个数写入 CSV
                csv_writer.writerow([image_id, apple_count])

                # 存储苹果个数和ID号到列表中
                apple_count_list.append((int(image_id), apple_count))

        # 绘制苹果图像ID号的分布直方图
        if apple_count_list:
            image_ids, apple_counts = zip(*apple_count_list)
            plt.bar(image_ids, apple_counts)
            plt.xlabel('图片ID号')
            plt.ylabel('苹果个数')
            plt.title('苹果图像ID号的分布直方图')
            plt.show()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
