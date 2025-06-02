from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.train(data='Dataset/SplitData/data.yaml', epochs=3)

if __name__ == '__main__':
    main()
