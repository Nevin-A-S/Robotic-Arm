import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSDLiteHead
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.transforms import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
from PIL import Image


# ==================== DATASET PREPARATION ====================

class DetectionDataset(Dataset):
    def __init__(self, img_dir, annot_dir, transform=None):
        """
        Custom dataset for object detection
        Args:
            img_dir: Directory with images
            annot_dir: Directory with annotation files (assumed to be in PASCAL VOC format)
            transform: Optional transform to be applied
        """
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transform = transform
        
        # Get list of all images
        self.imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                    if f.endswith('.jpg') or f.endswith('.png')]
        self.imgs.sort()
        
        # Create class mapping (adjust these classes to match your dataset)
        self.class_names = ["background", "person", "car", "bicycle", "dog", "cat"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotation file path (assumes same name but different extension)
        ann_file = os.path.join(self.annot_dir, 
                               os.path.basename(img_path).replace('.jpg', '.xml').replace('.png', '.xml'))
        
        # Parse annotation (simplified - in real implementation, use proper XML parsing)
        boxes, labels = self._parse_voc_xml(ann_file)
        
        if self.transform:
            transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        # Convert to tensor format expected by SSD
        target = {}
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            
        target["boxes"] = boxes
        target["labels"] = labels
        
        return img, target
    
    def _parse_voc_xml(self, xml_file):
        """
        Parse PASCAL VOC XML file
        This is a simplified version - in real implementation use a proper XML parser
        """
        # Placeholder for actual XML parsing logic
        # In real implementation, use xml.etree.ElementTree or similar
        
        # Mock implementation - replace with actual parsing
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in self.class_to_idx:
                label = self.class_to_idx[class_name]
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Normalize coordinates (expected by albumentations)
                img_width = float(root.find('size').find('width').text)
                img_height = float(root.find('size').find('height').text)
                
                xmin /= img_width
                xmax /= img_width
                ymin /= img_height
                ymax /= img_height
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
        
        return boxes, labels

# Data augmentation and preprocessing
def get_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(320, 320),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(320, 320),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


# ==================== MODEL DEFINITION ====================

def create_mobilenet_ssd(num_classes):
    """Create a MobileNetV3-SSDLite model with custom number of classes"""
    # Create a backbone with pretrained weights
    backbone = mobilenet_backbone(
        "mobilenet_v3_large", 
        pretrained=True, 
        fpn=False,
        trainable_layers=3
    )
    
    # Specify the anchor generator parameters
    anchor_generator = torchvision.models.detection.anchor_utils.DefaultBoxGenerator(
        aspect_ratios=[[2, 3] for _ in range(6)],
        min_ratio=0.2,
        max_ratio=0.95
    )
    
    # Create the SSD model
    model = torchvision.models.detection.SSDLite320_MobileNet_V3_Large(
        num_classes=num_classes,
        pretrained_backbone=False,
        trainable_backbone_layers=3
    )
    
    # Use the created backbone
    model.backbone = backbone
    
    return model


# ==================== TRAINING CODE ====================

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    
    for images, targets in data_loader:
        # Move to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """Train the SSD model"""
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
                
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


# ==================== QUANTIZATION ====================

def quantize_model(model, calibration_loader, device):
    """
    Quantize the model using PyTorch's quantization
    
    Args:
        model: The trained model
        calibration_loader: DataLoader for calibration data
        device: Device to run on
    """
    # Set model to evaluation mode
    model.eval()
    
    # Prepare for quantization
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    model_fp32_prepared = torch.quantization.prepare(model)
    
    # Calibrate with representative dataset
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = [image.to(device) for image in images]
            model_fp32_prepared(images)
    
    # Convert to quantized model
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    
    return model_int8


# ==================== RASPBERRY PI INFERENCE ====================

class OptimizedDetector:
    def __init__(self, model_path, device='cpu', confidence_threshold=0.5, class_names=None):
        """
        Initialize the optimized detector for Raspberry Pi
        
        Args:
            model_path: Path to the quantized model
            device: Device to run inference on
            confidence_threshold: Confidence threshold for detections
            class_names: List of class names
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names or ["background", "person", "car", "bicycle", "dog", "cat"]
        
        # Load the quantized model
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # Define preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        # Convert BGR to RGB (if using OpenCV)
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def detect(self, image):
        """
        Run detection on an image
        
        Args:
            image: The input image (numpy array or PIL Image)
            
        Returns:
            List of detected objects with bounding boxes, class labels, and confidence scores
        """
        # Get original image dimensions for scaling bounding boxes
        if isinstance(image, np.ndarray):
            orig_height, orig_width = image.shape[:2]
        else:
            orig_width, orig_height = image.size
        
        # Preprocess the image
        input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            predictions = self.model(input_tensor)
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


# ==================== RASPBERRY PI LIVE INFERENCE ====================

def run_live_inference(model_path, source=0, display=True):
    """
    Run live inference with webcam or video file
    
    Args:
        model_path: Path to the quantized model
        source: Camera index or video file path
        display: Whether to display output
    """
    # Initialize video capture
    cap = cv2.VideoCapture(source)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Initialize detector
    detector = OptimizedDetector(
        model_path=model_path,
        confidence_threshold=0.5
    )
    
    # For calculating FPS
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    print("Starting live inference. Press 'q' to quit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Run detection
        detections, inference_time = detector.detect(frame)
        
        # Calculate FPS
        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        # Draw results
        if display:
            # Draw bounding boxes and labels
            for detection in detections:
                box = detection['box']
                score = detection['score']
                class_name = detection['class_name']
                
                xmin, ymin, xmax, ymax = box
                
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {score:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display FPS and inference time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Object Detection', frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# ==================== OPTIMIZATION TECHNIQUES ====================

def optimize_for_raspberry_pi(model):
    """
    Apply additional optimizations for Raspberry Pi
    """
    # 1. Use JIT to compile the model
    # This creates an optimized and serializable version of the model
    model_jit = torch.jit.script(model)
    
    # 2. Optimize for inference (freeze weights and fuse layers)
    model_jit = torch.jit.optimize_for_inference(model_jit)
    
    return model_jit


# ==================== MAIN TRAINING & DEPLOYMENT PIPELINE ====================

def main():
    # Define paths
    train_img_dir = "path/to/train/images"
    train_annot_dir = "path/to/train/annotations"
    val_img_dir = "path/to/val/images"
    val_annot_dir = "path/to/val/annotations"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define class names (modify according to your dataset)
    class_names = ["background", "person", "car", "bicycle", "dog", "cat"]
    num_classes = len(class_names)
    
    # Create datasets
    train_dataset = DetectionDataset(
        img_dir=train_img_dir,
        annot_dir=train_annot_dir,
        transform=get_transform(train=True)
    )
    
    val_dataset = DetectionDataset(
        img_dir=val_img_dir,
        annot_dir=val_annot_dir,
        transform=get_transform(train=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch)),
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda batch: tuple(zip(*batch)),
        num_workers=4
    )
    
    # Create calibration loader for quantization (subset of validation data)
    calibration_dataset = torch.utils.data.Subset(val_dataset, indices=range(min(100, len(val_dataset))))
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: tuple(zip(*batch)),
        num_workers=1
    )
    
    # Create model
    model = create_mobilenet_ssd(num_classes=num_classes)
    model.to(device)
    
    # Train model
    print("Starting training...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=15
    )
    
    # Save trained model
    torch.save(trained_model.state_dict(), "mobilenet_ssd_trained.pth")
    print("Trained model saved.")
    
    # Prepare for quantization
    trained_model.to('cpu')
    trained_model.eval()
    
    # Quantize model
    print("Quantizing model...")
    quantized_model = quantize_model(
        model=trained_model,
        calibration_loader=calibration_loader,
        device='cpu'
    )
    
    # Additional optimizations
    print("Applying additional optimizations...")
    optimized_model = optimize_for_raspberry_pi(quantized_model)
    
    # Save optimized model
    optimized_model_path = "mobilenet_ssd_optimized.pt"
    torch.jit.save(optimized_model, optimized_model_path)
    print(f"Optimized model saved to {optimized_model_path}")
    
    print("Training and optimization complete!")
    
    # Deploy instructions
    print("""
    To run on Raspberry Pi:
    1. Transfer the optimized model file to the Pi
    2. Install requirements:
       - PyTorch (lightweight version for Pi)
       - OpenCV
       - NumPy
    3. Run the live inference script
    """)


# ==================== RASPBERRY PI DEPLOYMENT SCRIPT ====================

def raspi_deployment():
    """
    Script to be run on Raspberry Pi for live inference
    """
    # Set model path
    model_path = "mobilenet_ssd_optimized.pt"
    
    # Use Raspberry Pi camera or USB webcam
    camera_source = 0  # Use 0 for first connected camera
    
    # Run live inference
    run_live_inference(
        model_path=model_path,
        source=camera_source,
        display=True
    )


if __name__ == "__main__":
    # For training and optimization on development machine
    # main()
    
    # For deployment on Raspberry Pi
    # raspi_deployment()
    pass