import os
import cv2
import glob
import serial
import numpy as np

# Python3 program for the above approach
from math import atan

# Function to find the
# angle between two lines
def findAngle(M1, M2):
	PI = 3.14159265
	
	# Store the tan value of the angle
	angle = abs((M2 - M1) / (1 + M1 * M2))

	# Calculate tan inverse of the angle
	ret = atan(angle)

	# Convert the angle from
	# radian to degree
	val = (ret * 180) / PI

	# Print the result
	return (round(val, 4))


def calibrate_camera():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # arrays to store object points and image points from all the images
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    calib_path = 'calib_images/checkerboard'
    
    if not os.path.exists(calib_path):
        print(f"Warning: Calibration directory '{calib_path}' not found.")
        print("Using default camera parameters.")
  
        mtx = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
        dist = np.zeros((1, 5))
        return mtx, dist
    

    images = glob.glob(os.path.join(calib_path, '*.jpg'))
    
    if len(images) == 0:
        print("No calibration images found.")
        print("Using default camera parameters.")

        mtx = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
        dist = np.zeros((1, 5))
        return mtx, dist
    
    image_size = None
    
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Could not read image: {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # Store for later use
        
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        
        if ret:
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    
    if len(objpoints) == 0:
        print("No chessboard patterns found in the images.")
        print("Using default camera parameters.")

        mtx = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
        dist = np.zeros((1, 5))
        return mtx, dist
    

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    print("Camera calibrated successfully.")
    return mtx, dist

def setServoAngle(ser, index, angle):
    command = f"SET,{index},{angle}\n"
    ser.write(command.encode())

def main():
    #Starting Serial Connection with Arduino at COM3 at 9600 baud rate
    # ser = serial.Serial('COM3', 9600)
    # ser.flush() #Clearing the serial buffer # Set Base Motor to forward position
    mtx, dist = calibrate_camera()
    index = 3
    angle = 140
    command = f"SET,{index},{angle}\n"
    # ser.write(command.encode())
    # ser.flush()

    cap = cv2.VideoCapture("http://192.168.0.176:81/stream")
    # cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        # if ser.in_waiting > 0:
        #     serial_data = ser.readline().decode('utf-8').strip()
        #     print("Received from Arduino:", serial_data)
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i in range(len(corners)):
                marker_corners = corners[i].reshape((4, 2))
                
                marker_corners = marker_corners.astype(int)
                
                top_left = marker_corners[0]
                top_right = marker_corners[1]
                center = (top_left + top_right) // 2

                cv2.line(frame, (center[0],0), (center[0],1280), (0, 255, 0), 2)

                #Slope of deviation Line

                slope = (center[1] - 1024) / (center[0] - 1280//2)

                print(findAngle(slope, 1))

                
                #convert coordinat to servo degree
                # servoX = np.interp(center[0], [0, 1280], [0, 180])

                cv2.line(frame, tuple(center), (1280//2,1024), (0, 0, 255), 2)
                # if center[0] > 1280//2:
                #     angle -= 1 
                #     setServoAngle(ser, 3, angle)
                # elif center[0] < 1280//2:
                #     angle += 1
                #     setServoAngle(ser, 3, angle)
                # elif center[0] == 1280//2:
                #     print("Centered")

                if ids is not None:
                    marker_id = ids[i][0]
                    cv2.putText(frame, f"ID: {marker_id}", 
                                (top_left[0], top_left[1] - 10),
                                font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                
                marker_size = 0.05 
                objPoints = np.array([
                    [-marker_size/2, marker_size/2, 0],
                    [marker_size/2, marker_size/2, 0],
                    [marker_size/2, -marker_size/2, 0],
                    [-marker_size/2, -marker_size/2, 0]
                ])
                
                success, rvec, tvec = cv2.solvePnP(
                    objPoints, 
                    corners[i].reshape((4, 2)), 
                    mtx, 
                    dist
                )
                
                if success:
                    cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)
        else:

            cv2.putText(frame, "No Markers Detected", (10, 30), 
                        font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.line(frame, (1280//2,0), (1280//2,1024), (0, 0, 255), 2)
        cv2.imshow('ArUco Marker Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # ser.close()
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()