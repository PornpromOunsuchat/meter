from ultralytics import YOLO
import cv2

# Load models
region_model = YOLO("models/region.pt")
house_model = YOLO("models/house_digit.pt")
meter_model = YOLO("models/meter_digit.pt")


def read_digits(model, crop):
    results = model(crop)[0]

    digits = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])

        digits.append((int(x1), str(cls)))

    # sort left → right
    digits = sorted(digits, key=lambda x: x[0])

    return "".join([d[1] for d in digits])


def process_image(image_path):
    img = cv2.imread(image_path)

    results = region_model(img)[0]

    house_number = ""
    meter_number = ""

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = img[y1:y2, x1:x2]

        # 🔥 KEY PART: choose model based on region
        if cls == 0:  # house_number
            house_number = read_digits(house_model, crop)

        elif cls == 1:  # meter_number
            meter_number = read_digits(meter_model, crop)

    return house_number, meter_number

from ultralytics import YOLO
import cv2
import os
import csv

# ======================
# LOAD MODELS (GPU AUTO)
# ======================
region_model = YOLO("models/region.pt")
house_model = YOLO("models/house_digit.pt")
meter_model = YOLO("models/meter_digit.pt")


# ======================
# READ DIGITS
# ======================
def read_digits(model, crop):

    results = model(crop, verbose=False)[0]

    digits = []

    if results.boxes is None:
        return ""

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])

        digits.append((int(x1), str(cls)))

    # sort digits left → right
    digits = sorted(digits, key=lambda x: x[0])

    return "".join([d[1] for d in digits])


# ======================
# PROCESS ONE IMAGE
# ======================
def process_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print(f"Cannot read {image_path}")
        return "", ""

    results = region_model(img, verbose=False)[0]

    house_number = ""
    meter_number = ""

    if results.boxes is None:
        return "", ""

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = img[y1:y2, x1:x2]

        # choose model by region
        if cls == 0:
            house_number = read_digits(house_model, crop)

        elif cls == 1:
            meter_number = read_digits(meter_model, crop)

    return house_number, meter_number


# ======================
# PROCESS FOLDER
# ======================
def process_folder(folder):

    results_list = []

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"Found {len(images)} images")

    for img_name in images:

        path = os.path.join(folder, img_name)

        house, meter = process_image(path)

        print(f"{img_name} → House:{house} Meter:{meter}")

        results_list.append([img_name, house, meter])

    return results_list


# ======================
# SAVE CSV
# ======================
def save_csv(results, filename="result.csv"):

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "House Number", "Meter Number"])
        writer.writerows(results)

    print(f"\nSaved → {filename}")


# ======================
# RUN
# ======================
if __name__ == "__main__":

    image_folder = "test_images"   # <-- put images here

    results = process_folder(image_folder)

    save_csv(results)


# # ===== RUN =====
# if __name__ == "__main__":
#     img_path = "test2.jpg"

#     house, meter = process_image(img_path)

#     print("House Number:", house)
#     print("Meter Number:", meter)