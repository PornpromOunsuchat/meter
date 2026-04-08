from ultralytics import YOLO
import torch


def main():

    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/house_digits/data.yaml",
        epochs=60,
        imgsz=320,      # digits don't need big resolution
        batch=32,       # adjust if VRAM small
        device=0,       # ✅ FORCE GPU
        workers=0,      # ✅ Windows safe
        amp=True,       # ✅ mixed precision (faster)
        cache=True      # ✅ faster loading
    )


if __name__ == "__main__":
    main()