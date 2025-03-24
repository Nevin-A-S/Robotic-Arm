import cv2
import torch

# 1. Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4

# 2. Open the video stream
stream_url = 'http://192.168.0.176:81/stream'
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open video stream.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # ---------------------
    # A. Mild Denoising
    # ---------------------
    # Median blur (3x3) to reduce speckle noise
    # denoised = cv2.medianBlur(frame, 3)

    # ---------------------
    # B. Contrast Enhancement
    # ---------------------
    # Convert to LAB and apply CLAHE to L channel
    # lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # l_clahe = clahe.apply(l)
    # lab_clahe = cv2.merge((l_clahe, a, b))
    # enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # # ---------------------
    # # C. Mild Sharpening
    # # ---------------------
    # # Unsharp mask technique
    # #  - Weighted sum: result = 1.2*enhanced - 0.2*gaussian_blur
    # #  - Adjust these multipliers if you want more or less sharpening
    # blurred = cv2.GaussianBlur(enhanced, (0, 0), 2)
    # sharpened = cv2.addWeighted(enhanced, 1.2, blurred, -0.2, 0)

    # ---------------------
    # D. YOLO Inference
    # ---------------------
    results = model(frame)
    annotated_frame = results.render()[0]

    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
