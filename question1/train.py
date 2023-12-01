from ultralytics import YOLO
import multiprocessing

def main():
    # 加载模型
    model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）

    # 训练模型
    model.train(data='./final/question1/Apple Vision.v9-applevision-larger-set-80-20-10-split.yolov8/data.yaml', epochs=10, imgsz=640 ,resume=False)

    # 保存训练后的模型为'apple.pt'
    model.export(format='onnx')
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

