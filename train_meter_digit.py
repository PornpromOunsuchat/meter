from ultralytics import YOLO
import torch


def main():

    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/meter_digits/data.yaml",
        epochs=100,
        imgsz=320,
        batch=32,
        device=0,       # ✅ GPU
        workers=0,      # ✅ Windows fix
        amp=True,
        cache=True
    )


if __name__ == "__main__":
    main()