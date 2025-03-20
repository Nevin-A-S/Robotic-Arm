import cv2

stream_url = 'http://192.168.0.176:81/stream'  # Replace <ESP32_IP> with your camera's IP address

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Unable to open video stream. Check the URL and network connection.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    cv2.imshow("Cam Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()