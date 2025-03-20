import cv2
import urllib.request
import numpy as np

# Use the correct direct image URL
url = 'https://upload.wikimedia.org/wikipedia/commons/3/3f/JPEG_example_flower.jpg'
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1)  # Load it as it is

# Check if the image was loaded successfully
if img is None:
    print("Failed to load image. Check the URL.")
else:
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
