if __name__ == '__main__':
    from ultralytics import YOLO
    model = YOLO('Yolo-Weights/yolov8n.pt')
    model.train(data='config.yaml', epochs=100, imgsz=640, device='cpu')

    