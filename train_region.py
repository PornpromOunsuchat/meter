from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/region/data.yaml",
        epochs=50,
        imgsz=640,
        workers=0   # optional but safest on Windows
    )

if __name__ == "__main__":
    main()