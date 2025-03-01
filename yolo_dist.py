import cv2
from ultralytics import solutions

# Load the image from file
image_path = "http://192.168.0.176/capture?_cb=1740332996243"  # Replace with your image file path
im0 = cv2.imread(image_path)
assert im0 is not None, "Error loading image. Please check the image path."

# Initialize the distance-calculation object
distance = solutions.DistanceCalculation(model="yolo11n.pt", show=True)

# Process the image for distance calculation
result = distance.calculate(im0)

# Optionally display the resulting image
cv2.imshow("Distance Calculation", result)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()

# Optionally, save the processed image to a file
cv2.imwrite("distance_calculation_result.jpg", result)
