import cv2
import base64
import time
import json
from ollama import chat

# URL for your IP camera
camera_url = "http://192.168.0.176:81/stream"

# Open the video capture (adjust parameters if needed for your camera)
cap = cv2.VideoCapture(camera_url)
if not cap.isOpened():
    print("Error: Could not open video stream from the camera.")
    exit(1)

print("Starting frame processing. Press Ctrl+C to stop.")

while True:
    start_time = time.time()

    # Capture one frame from the stream
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from the camera.")
        break

    # Encode the captured frame as JPEG
    ret_enc, buffer = cv2.imencode('.jpg', frame)
    if not ret_enc:
        print("Error: Could not encode the frame to JPEG.")
        continue
    img_bytes = buffer.tobytes()

    # Convert the JPEG bytes to base64 string
    img_data = base64.b64encode(img_bytes).decode("utf-8")

    # Define the prompt: instruct the model to return a JSON with only objects and distances.
    prompt = (
        "Analyze the image and output a JSON object with exactly two keys: "
        "'objects' (a list of detected objects) and 'distances' (a list of their estimated distances from the camera in meters). "
        "Return only this JSON object with no extra text."
    )

    # Send the request to the llava-phi3 model with the image data
    response = chat(
        model='llava-phi3',
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [img_data],
            }
        ],
    )

    # Measure processing time for the frame
    processing_time = time.time() - start_time

    # Try parsing the response as JSON; if not possible, keep the raw output.
    try:
        result = json.loads(response.message.content)
    except Exception as e:
        result = {"raw_output": response.message.content}

    # Print the structured output along with the time taken for this frame.
    print(f"\nFrame processed in {processing_time:.2f} seconds:")
    print(json.dumps(result, indent=2))

    # Ensure that we process one frame per second
    elapsed = time.time() - start_time
    if elapsed < 1:
        time.sleep(1 - elapsed)

# Release the capture resource
cap.release()
