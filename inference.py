import cv2
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class QuantizedDetector:
    def __init__(self, model_path, class_names=None, confidence_threshold=0.5):
        """Initialize the detector with the quantized model"""
        self.device = torch.device('cpu')
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names or ["background", "degradable", "non degradable"]
        
        # Load the quantized model
        print(f"Loading model from {model_path}")
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # Create preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_image(self, image):
        """Convert OpenCV image to PyTorch tensor"""
        # Convert BGR to RGB
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def detect(self, image):
        """Run detection on an image"""
        # Save original dimensions for scaling bounding boxes
        if isinstance(image, np.ndarray):
            orig_height, orig_width = image.shape[:2]
        else:
            orig_width, orig_height = image.size
        
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            predictions = self.model([input_tensor[0]])
        inference_time = time.time() - start_time
        
        # Process predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Scale boxes to original image size
        scaled_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            xmin = int(xmin * orig_width / 320)
            xmax = int(xmax * orig_width / 320)
            ymin = int(ymin * orig_height / 320)
            ymax = int(ymax * orig_height / 320)
            scaled_boxes.append([xmin, ymin, xmax, ymax])
        
        results = []
        for box, score, label in zip(scaled_boxes, scores, labels):
            class_name = self.class_names[label]
            results.append({
                'box': box,
                'score': float(score),
                'class_name': class_name,
                'class_id': int(label)
            })
        
        return results, inference_time

def open_camera_with_different_backends(source):
    """Try opening the camera with different backends"""
    # List of available backends in OpenCV
    backends = [
        cv2.CAP_DSHOW,      # DirectShow (Windows)
        cv2.CAP_MSMF,       # Media Foundation (Windows)
        cv2.CAP_ANY,        # Auto-detect
        cv2.CAP_V4L2,       # Video for Linux
        cv2.CAP_GSTREAMER   # GStreamer
    ]
    
    for backend in backends:
        print(f"Trying to open camera with backend {backend}")
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"Successfully opened camera with backend {backend}")
                return cap
            else:
                print(f"Camera opened with backend {backend} but couldn't read frame")
                cap.release()
    
    # If all backends fail, try a simpler approach as last resort
    print("Trying simple camera open method...")
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap
    
    return None

def run_inference(model_path, source=0, confidence=0.5):
    """Run inference with camera or video"""
    # Initialize detector
    detector = QuantizedDetector(
        model_path=model_path,
        confidence_threshold=confidence
    )
    
    # Open video capture with different backends if source is a camera
    print(f"Opening video source: {source}")
    if isinstance(source, int):
        # For camera, try different backends
        cap = open_camera_with_different_backends(source)
    else:
        # For video files, use standard approach
        cap = cv2.VideoCapture(source)
    
    if not cap or not cap.isOpened():
        print("Error: Could not open video source")
        print("Please check if camera is connected or try a different camera index (--source 1, 2, etc.)")
        return
    
    # Show camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {width}x{height} at {fps}fps")
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    print("Starting inference. Press 'q' to quit.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            # Try to reconnect
            print("Trying to reconnect...")
            if isinstance(source, int):
                cap = open_camera_with_different_backends(source)
            else:
                cap = cv2.VideoCapture(source)
                
            if not cap or not cap.isOpened():
                print("Reconnect failed, exiting")
                break
            continue
        
        # Run detection
        try:
            detections, inference_time = detector.detect(frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:
                end_time = time.time()
                fps_display = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Draw results
            for detection in detections:
                box = detection['box']
                score = detection['score']
                class_name = detection['class_name']
                
                xmin, ymin, xmax, ymax = box
                
                # Draw bounding box with color based on class
                if 'degradable' in class_name.lower():
                    color = (0, 255, 0)  # Green for degradable
                else:
                    color = (0, 0, 255)  # Red for non-degradable
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Draw label with confidence score
                label = f"{class_name}: {score:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display FPS and inference time
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow('Object Detection', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error during detection: {e}")
            continue
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def try_dummy_camera():
    """Create a dummy camera feed if no real camera available"""
    print("Creating dummy video feed for testing...")
    width, height = 640, 480
    dummy_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw some shapes to make it interesting
    cv2.rectangle(dummy_img, (100, 100), (300, 300), (0, 255, 0), -1)
    cv2.circle(dummy_img, (450, 250), 100, (0, 0, 255), -1)
    cv2.putText(dummy_img, "Dummy Feed", (width//2 - 70, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return dummy_img

def image_inference(model_path, image_path, confidence=0.5):
    """Run inference on a single image file"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Initialize detector
    detector = QuantizedDetector(
        model_path=model_path,
        confidence_threshold=confidence
    )
    
    # Run detection
    detections, inference_time = detector.detect(image)
    print(f"Inference time: {inference_time*1000:.1f}ms")
    
    # Draw results
    for detection in detections:
        box = detection['box']
        score = detection['score']
        class_name = detection['class_name']
        
        xmin, ymin, xmax, ymax = box
        
        # Draw bounding box with color based on class
        if 'degradable' in class_name.lower():
            color = (0, 255, 0)  # Green for degradable
        else:
            color = (0, 0, 255)  # Red for non-degradable
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image, label, (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Show results
    cv2.imshow('Detection Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save output image
    output_path = f"output_{image_path.split('/')[-1]}"
    cv2.imwrite(output_path, image)
    print(f"Result saved to {output_path}")

def list_available_cameras():
    """Try to list and test available camera indices"""
    print("Checking for available cameras...")
    available_cameras = []
    
    # Try first 10 camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera index {i} is available")
                available_cameras.append(i)
            cap.release()
    
    if available_cameras:
        print(f"Found {len(available_cameras)} camera(s): {available_cameras}")
        print(f"Try using --source {available_cameras[0]} to use the first available camera")
    else:
        print("No cameras found")
    
    return available_cameras

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with quantized model")
    parser.add_argument("--model", type=str, default="mobilenet_ssd_optimized.pt",
                       help="Path to the quantized model file")
    parser.add_argument("--source", type=str, default="0",
                       help="Source (0 for webcam, path for video file or image)")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold for detections")
    parser.add_argument("--image", action="store_true",
                       help="Set this flag if source is an image file")
    parser.add_argument("--list-cameras", action="store_true",
                       help="List available camera indices and exit")
    parser.add_argument("--dummy", action="store_true",
                       help="Use dummy video feed for testing without camera")
    
    args = parser.parse_args()
    
    # List cameras if requested
    if args.list_cameras:
        list_available_cameras()
        exit()
    
    # Determine if source is numeric (camera index)
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # Check if model file exists
    import os
    if not os.path.exists(args.model):
        print(f"Warning: Model file {args.model} not found!")
        proceed = input("Do you want to continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Exiting.")
            exit()
    
    # Run inference
    if args.image:
        print(f"Running inference on image: {source}")
        if os.path.exists(source):
            image_inference(args.model, source, args.confidence)
        else:
            print(f"Error: Image file {source} not found")
    elif args.dummy:
        print("Using dummy video feed for testing")
        # Use a fixed dummy image for visualization
        dummy_img = try_dummy_camera()
        detector = QuantizedDetector(args.model, confidence_threshold=args.confidence)
        while True:
            cv2.imshow('Dummy Feed', dummy_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        print(f"Running inference with source: {source}")
        run_inference(args.model, source, args.confidence) 