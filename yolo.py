import cv2
import torch

# 1. Load the YOLOv5 model from PyTorch Hub (this may download the model on first run)
# model = torch.hub.load('ultralytics/yolov5s', 'yolov5', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# Optionally, set model parameters (e.g., confidence threshold) like:
# model.conf = 0.4

# 2. Define the URL of your ESP32-CAM MJPEG stream.

stream_url = 'http://192.168.0.176:81/stream'  # Replace <ESP32_IP> with your camera's IP address

# 3. Open the video stream.
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Unable to open video stream. Check the URL and network connection.")
    exit()

print("Press 'q' to quit.")

while True:
    # 4. Read a frame from the stream.
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    # 5. Run YOLO object detection on the frame.
    results = model(frame)
    print('Result Fetched')

    # 6. Render the detection results onto the frame.
    # results.render() returns a list of images (one per input image). For a single image, use the first element.
    annotated_frame = results.render()[0]

    # 7. Display the annotated frame.
    cv2.imshow("YOLO Object Detection", annotated_frame)
    print(annotated_frame)

    # 8. Exit when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Cleanup: release the capture and close OpenCV windows.
cap.release()
cv2.destroyAllWindows()