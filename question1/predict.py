from ultralytics import YOLO
import multiprocessing
from PIL import Image
import os
import csv
import matplotlib.pyplot as plt

def plot_apple_positions(apple_positions, image_name, output_folder='./pic/scatter/'):
    # 将坐标拆分成两个列表，分别存储 x 和 y 坐标
    x_coordinates = [box[0] for box in apple_positions]
    y_coordinates = [180 - box[1] for box in apple_positions]  # 调整坐标系

    # 绘制散点图
    plt.scatter(x_coordinates, y_coordinates, marker='o', color='red')
    plt.title('Apple Positions')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # 设置坐标轴的范围
    plt.xlim(0, 270)  # 横坐标范围
    plt.ylim(0, 180)  # 纵坐标范围

    # 保存散点图
    scatter_file_path = os.path.join(output_folder, f'{image_name}.png')
    plt.savefig(scatter_file_path)

    plt.show()

def main():
    # 加载模型
    model = YOLO('./runs/detect/train5/weights/best.onnx')  # 加载官方模型

    # 图片文件夹路径
    image_folder = './Attachment 1/'

    # 检测结果保存文件夹路径
    output_folder = './runs/predict/'
    os.makedirs(output_folder, exist_ok=True)

    # 获取图片文件列表
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 用于保存结果的 CSV 文件
    csv_file_path = os.path.join(output_folder, 'count.csv')


    with open(csv_file_path, 'w', newline='') as csvfile:
        # 定义 CSV 写入对象
        csv_writer = csv.writer(csvfile)

        # 写入 CSV 头部
        csv_writer.writerow(['图片序号', '苹果个数'])
        # 循环处理每张图片
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_folder, image_file)
            image_name = os.path.splitext(image_file)[0]
        # 使用模型进行预测
            results = model.predict(image_path, save=False)
        # 查看 Results 对象的属性
        # 打印所有属性
        # 获取检测到的苹果个数

            print("苹果个数:", len(results[0].boxes))
        # 保存绘制的图
        # 将图片序号和目标个数写入 CSV
            csv_writer.writerow([image_name, len(results[0].boxes)])
        # 提取苹果位置
        apple_positions = [(box[0], box[1]) for box in results[0].boxes.xywh]
        # 绘制二维散点图
        plot_apple_positions(apple_positions, image_name)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()