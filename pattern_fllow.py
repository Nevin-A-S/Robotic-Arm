import numpy as np
import cv2
import glob
import os

def calibrate_camera():
    # termination criteria for the iterative algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points for a 7x6 checkerboard
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # arrays to store object points and image points from all the images
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    # path to calibration images
    calib_path = 'calib_images/checkerboard'
    
    # check if calibration directory exists
    if not os.path.exists(calib_path):
        print(f"Warning: Calibration directory '{calib_path}' not found.")
        print("Using default camera parameters.")
        # Default camera matrix and distortion coefficients
        mtx = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
        dist = np.zeros((1, 5))
        return mtx, dist
    
    # get all jpg files in the calibration directory
    images = glob.glob(os.path.join(calib_path, '*.jpg'))
    
    if len(images) == 0:
        print("No calibration images found.")
        print("Using default camera parameters.")
        # Default camera matrix and distortion coefficients
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
        
        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        
        if ret:
            objpoints.append(objp)
            
            # refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    
    if len(objpoints) == 0:
        print("No chessboard patterns found in the images.")
        print("Using default camera parameters.")
        # Default camera matrix and distortion coefficients
        mtx = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
        dist = np.zeros((1, 5))
        return mtx, dist
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    print("Camera calibrated successfully.")
    return mtx, dist

def main():
    # Get camera calibration parameters
    mtx, dist = calibrate_camera()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # For OpenCV 4.11
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    # Font for displaying text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is not captured properly
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # Draw detected markers
        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Loop through all detected markers
            for i in range(len(corners)):
                # Get the corner points of the marker
                marker_corners = corners[i].reshape((4, 2))
                
                # Convert to int (for pixel coordinates)
                marker_corners = marker_corners.astype(int)
                
                # Get top-left corner for text placement
                top_left = marker_corners[0]
                
                # Draw marker ID
                if ids is not None:
                    marker_id = ids[i][0]
                    cv2.putText(frame, f"ID: {marker_id}", 
                                (top_left[0], top_left[1] - 10),
                                font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Calculate marker pose
                # Note: In OpenCV 4.11, we use SolvePnP directly instead of estimatePoseSingleMarkers
                # We need to define 3D points for a marker of known size
                marker_size = 0.05  # 5cm marker size
                objPoints = np.array([
                    [-marker_size/2, marker_size/2, 0],
                    [marker_size/2, marker_size/2, 0],
                    [marker_size/2, -marker_size/2, 0],
                    [-marker_size/2, -marker_size/2, 0]
                ])
                
                # Using solvePnP to get pose
                success, rvec, tvec = cv2.solvePnP(
                    objPoints, 
                    corners[i].reshape((4, 2)), 
                    mtx, 
                    dist
                )
                
                if success:
                    # Draw coordinate axes
                    cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)
        else:
            # Display 'No Markers' message
            cv2.putText(frame, "No Markers Detected", (10, 30), 
                        font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display the resulting frame
        cv2.imshow('ArUco Marker Tracking', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()