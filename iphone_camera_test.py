import cv2
import numpy as np
import time

# DroidCam typically uses this format - adjust IP and port as needed
# Format: http://IP:PORT/video
ip = "192.168.0.111"  # Replace with your iPhone's IP address
port = "4747"  # Default DroidCam port
url = f"http://{ip}:{port}/video"

# Create a VideoCapture object
cap = cv2.VideoCapture(url)

# Check if camera opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video stream from {url}")
    print("Please check that:")
    print("1. DroidCam is running on your iPhone")
    print("2. Your computer and iPhone are on the same network")
    print("3. The IP address and port are correct")
    exit()

# Add error handling and retry mechanism
max_retries = 5
retry_count = 0

while True:
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly ret is True
        if not ret:
            retry_count += 1
            print(f"Error reading frame. Retry {retry_count}/{max_retries}")
            
            if retry_count >= max_retries:
                print("Max retries reached. Reconnecting to camera...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(url)
                retry_count = 0
                
            time.sleep(0.5)
            continue
        
        # Reset retry counter on successful frame
        retry_count = 0
        
        # Display the resulting frame
        cv2.imshow('iPhone Camera (DroidCam)', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()