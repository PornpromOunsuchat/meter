# from ultralytics import YOLO
# import cv2
# import os

# region_model = YOLO("models/region.pt")
# house_model = YOLO("models/house_digit.pt")
# meter_model = YOLO("models/meter_digit.pt")


# def read_meter(image_path, save_crop_folder):

#     img = cv2.imread(image_path)

#     if img is None:
#         return ("NOT FOUND", "NOT FOUND")

#     # -----------------------------
#     # HOUSE NUMBER
#     # -----------------------------
#     house_number = "NOT FOUND"

#     house_results = house_model(img)[0]

#     if house_results.boxes is not None and len(house_results.boxes) > 0:

#         digits = []

#         for box, cls in zip(
#             house_results.boxes.xyxy,
#             house_results.boxes.cls
#         ):
#             x = int(box[0])
#             digit = house_model.names[int(cls)]
#             digits.append((x, digit))

#         digits.sort(key=lambda x: x[0])
#         house_number = "".join([d[1] for d in digits])

#     # -----------------------------
#     # METER NUMBER
#     # -----------------------------
#     meter_number = "NOT FOUND"

#     meter_results = meter_model(img)[0]

#     if meter_results.boxes is not None and len(meter_results.boxes) > 0:

#         digits = []

#         for box, cls in zip(
#             meter_results.boxes.xyxy,
#             meter_results.boxes.cls
#         ):
#             x = int(box[0])
#             digit = meter_model.names[int(cls)]
#             digits.append((x, digit))

#         digits.sort(key=lambda x: x[0])
#         meter_number = "".join([d[1] for d in digits])

#     return (house_number, meter_number)

from ultralytics import YOLO
import cv2
import os

# ==============================
# Load models (only once)
# ==============================
region_model = YOLO("models/region.pt")
house_model = YOLO("models/house_digit.pt")
meter_model = YOLO("models/meter_digit.pt")


# ==============================
# Read digits helper
# ==============================
def read_digits(model, crop):

    results = model(crop)[0]

    digits = []

    if results.boxes is None:
        return ""

    for box in results.boxes:
        x1 = int(box.xyxy[0][0])
        cls = int(box.cls[0])

        digits.append((x1, str(cls)))

    # left → right order
    digits = sorted(digits, key=lambda x: x[0])

    return "".join([d[1] for d in digits])


# ==============================
# Main pipeline
# ==============================
def read_meter(image_path, save_crop_folder):

    img = cv2.imread(image_path)

    if img is None:
        print("Cannot read image:", image_path)
        return None

    os.makedirs(save_crop_folder, exist_ok=True)

    results = region_model(img)[0]

    house_number = ""
    meter_number = ""

    # ------------------------------
    # Detect regions
    # ------------------------------
    for box in results.boxes:

        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = img[y1:y2, x1:x2]

        # Save crop image
        crop_name = (
            os.path.splitext(os.path.basename(image_path))[0]
            + f"_region{cls}.jpg"
        )

        crop_path = os.path.join(save_crop_folder, crop_name)
        cv2.imwrite(crop_path, crop)

        # ------------------------------
        # Choose correct digit model
        # ------------------------------
        if cls == 0:  # house number region
            house_number = read_digits(house_model, crop)

        elif cls == 1:  # meter number region
            meter_number = read_digits(meter_model, crop)

    return house_number, meter_number