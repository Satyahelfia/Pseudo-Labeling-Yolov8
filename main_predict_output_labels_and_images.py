import os
from ultralytics import YOLO
import cv2

# Path ke model dan folder gambar
model_path = 'runs/detect/train6_roe_example/weights/best.pt'
image_folder = 'roe'
annotation_folder = 'roe_annotations/labels'
output_image_folder = 'roe_annotations/images'  # Folder untuk menyimpan gambar hasil deteksi

# Load model YOLOv8
model = YOLO(model_path)

# Fungsi untuk menyimpan anotasi hasil deteksi
def save_annotations(results, image_name):
    annotation_path = os.path.join(annotation_folder, f"{image_name}.txt")
    os.makedirs(annotation_folder, exist_ok=True)

    with open(annotation_path, 'w') as f:
        for result in results:
            boxes = result.boxes.xyxy  # Format [x_min, y_min, x_max, y_max]
            scores = result.boxes.conf  # Confidence score
            classes = result.boxes.cls  # Class label

            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
                cls = int(classes[i])
                # Format: class x_min y_min x_max y_max confidence
                f.write(f"{cls} {box[0]} {box[1]} {box[2]} {box[3]} {score}\n")

    print(f"Annotations saved to {annotation_path}")

# Fungsi untuk menggambar bounding box pada gambar
def draw_boxes(img, results):
    for result in results:
        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        classes = result.boxes.cls

        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            cls = int(classes[i])
            label = f"{cls} {score:.2f}"
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(img, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

# Fungsi untuk mendeteksi objek pada gambar
def detect_objects(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    # Simpan hasil anotasi
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    save_annotations(results, image_name)

    # Gambar bounding box pada gambar
    img_with_boxes = draw_boxes(img.copy(), results)

    # Simpan gambar hasil deteksi
    output_image_path = os.path.join(output_image_folder, f"{image_name}_detected.jpg")
    os.makedirs(output_image_folder, exist_ok=True)
    cv2.imwrite(output_image_path, img_with_boxes)
    print(f"Detected image saved to {output_image_path}")

# Loop untuk memproses semua gambar di folder road_sign
if __name__ == "__main__":
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Cek apakah file adalah gambar
            detect_objects(img_path)
