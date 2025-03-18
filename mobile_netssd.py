import os
import time
import numpy as np
import torch
# Add version check before importing torchvision
torch_version = torch.__version__
print(f"PyTorch version: {torch_version}")

# Try to import torchvision with error handling
try:
    import torchvision
    print(f"torchvision version: {torchvision.__version__}")
except RuntimeError as e:
    print(f"Error importing torchvision: {e}")
    print("Please install compatible versions of PyTorch and torchvision.")
    print("For PyTorch 2.0: pip install torchvision==0.15.0")
    print("For PyTorch 1.13: pip install torchvision==0.14.0")
    print("For PyTorch 1.12: pip install torchvision==0.13.0")
    exit(1)

import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.transforms import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2

# Try to import albumentations with error handling
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    print("Successfully imported albumentations")
except ImportError:
    print("Error: albumentations module not found.")
    print("Please install albumentations: pip install albumentations")
    print("Or with conda: conda install -c conda-forge albumentations")
    exit(1)

import copy
from PIL import Image


# ==================== DATASET PREPARATION ====================

class DetectionDataset(Dataset):
    def __init__(self, img_dir, annot_dir, transform=None, class_names=None):
        """
        Custom dataset for object detection
        Args:
            img_dir: Directory with images
            annot_dir: Directory with annotation files (assumed to be in PASCAL VOC format)
            transform: Optional transform to be applied
            class_names: List of class names (if None, defaults to ["background", "crack"])
        """
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transform = transform
        
        # Create class mapping
        self.class_names = class_names or ["background", "crack"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        # Get list of all images - handle nested directory structure
        self.imgs = []
        # Check if img_dir contains subdirectories
        subdirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
        
        if subdirs:
            # If there are subdirectories, look for images in each subdirectory
            print(f"Found {len(subdirs)} subdirectories in image directory")
            for subdir in subdirs:
                subdir_path = os.path.join(img_dir, subdir)
                for f in os.listdir(subdir_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.imgs.append(os.path.join(subdir_path, f))
        else:
            # If no subdirectories, look for images directly in img_dir
            self.imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        self.imgs.sort()
        print(f"Found {len(self.imgs)} images for dataset")
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotation file path - extract just the filename without directory or extension
        base_filename = os.path.basename(img_path)  # Get just the filename (e.g., "data201.jpg")
        base_name = os.path.splitext(base_filename)[0]  # Remove the extension (e.g., "data201")
        ann_file = os.path.join(self.annot_dir, f"{base_name}.xml")
        
        # Check if annotation file exists
        if not os.path.exists(ann_file):
            print(f"Warning: Annotation file {ann_file} does not exist for image {img_path}")
            # Return empty boxes and labels when annotation is missing
            boxes = []
            labels = []
        else:
            # Parse annotation
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
        """
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        # Get image dimensions from the XML file
        size_elem = root.find('size')
        if size_elem is not None:
            img_width = float(size_elem.find('width').text)
            img_height = float(size_elem.find('height').text)
        else:
            # Default to reasonable size if not found in XML
            print(f"Warning: No size info in {xml_file}, using default dimensions")
            img_width = 640.0
            img_height = 480.0
        
        for obj in root.findall('object'):
            # First check for the <n> tag which is used in the custom format
            class_name_elem = obj.find('n')
            if class_name_elem is None:
                # If <n> not found, check for standard <name> tag
                class_name_elem = obj.find('name')
                if class_name_elem is None:
                    print(f"Warning: No class name found in {xml_file} for object")
                    continue
            
            class_name = class_name_elem.text
            
            # Map class names to the format we expect
            if class_name == 'deg':
                class_name = 'degradable'
            elif class_name == 'non-deg':
                class_name = 'non degradable'
            
            if class_name in self.class_to_idx:
                label = self.class_to_idx[class_name]
                
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                    
                # Parse bounding box coordinates with error handling
                try:
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # Ensure coordinates are valid
                    if xmin >= xmax or ymin >= ymax:
                        print(f"Warning: Invalid bbox in {xml_file}: {xmin},{ymin},{xmax},{ymax}")
                        continue
                    
                    # For pascal_voc format, we need pixel coordinates
                    # No normalization needed - albumentations expects pixel coordinates
                    # Ensure values are within image dimensions
                    xmin = max(0, min(img_width, xmin))
                    xmax = max(0, min(img_width, xmax))
                    ymin = max(0, min(img_height, ymin))
                    ymax = max(0, min(img_height, ymax))
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)
                except (AttributeError, ValueError) as e:
                    print(f"Error parsing bounding box in {xml_file}: {e}")
            else:
                print(f"Warning: Unknown class '{class_name}' in {xml_file}")
        
        return boxes, labels

# Data augmentation and preprocessing
def get_transform(train):
    """
    Create a transform pipeline for the object detection task.
    
    Args:
        train: Whether this is for training (with augmentation) or not
        
    Returns:
        An albumentations Compose object
    """
    # For object detection, we need to specify the bbox_params
    bbox_params = A.BboxParams(
        format='pascal_voc',  # xmin, ymin, xmax, ymax in absolute pixels
        label_fields=['class_labels']  # specify which field has the labels
    )
    
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(320, 320),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=bbox_params)
    else:
        return A.Compose([
            A.Resize(320, 320),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=bbox_params)


# ==================== MODEL DEFINITION ====================

def create_mobilenet_ssd(num_classes):
    """Create a MobileNetV3-SSDLite model with custom number of classes"""
    # Create the SSD model with pretrained backbone
    model = ssdlite320_mobilenet_v3_large(
        pretrained=False,  # We don't want pretrained weights for the whole model
        num_classes=num_classes,
        pretrained_backbone=True  # But we do want pretrained backbone
    )
    
    # Alternatively, if you need more control over the backbone:
    # backbone = mobilenet_backbone(
    #     "mobilenet_v3_large", 
    #     pretrained=True, 
    #     fpn=False,
    #     trainable_layers=3
    # )
    # model.backbone = backbone
    
    return model


# ==================== TRAINING CODE ====================

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    valid_batches = 0
    
    for images, targets in data_loader:
        # Check if all targets have empty boxes
        all_empty = all(len(t["boxes"]) == 0 for t in targets)
        if all_empty:
            print("Warning: Skipping training batch with all empty annotations")
            continue
            
        # Move to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Check for NaN
        if torch.isnan(losses) or torch.isinf(losses):
            print("Warning: NaN or Inf loss detected. Skipping batch.")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        valid_batches += 1
    
    if valid_batches > 0:
        return total_loss / valid_batches
    else:
        print("Warning: No valid training batches found")
        return float('inf')  # Return a high loss value

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """Train the SSD model"""
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.01, weight_decay=5e-4)
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
        val_batches = 0
        model.train()  # Temporarily set to train mode to get losses
        with torch.no_grad():
            for images, targets in val_loader:
                # Check if all targets have empty boxes
                all_empty = all(len(t["boxes"]) == 0 for t in targets)
                if all_empty:
                    print("Warning: Skipping validation batch with all empty annotations")
                    continue
                
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # In train mode, the model returns a dict of losses
                loss_dict = model(images, targets)
                batch_loss = sum(loss for loss in loss_dict.values()).item()
                
                # Check for NaN values
                if not np.isnan(batch_loss) and not np.isinf(batch_loss):
                    val_loss += batch_loss
                    val_batches += 1
        
        # Calculate average val loss
        if val_batches > 0:
            val_loss /= val_batches
        else:
            # If no valid batches, set to a high value
            val_loss = float('inf')
            print("Warning: No valid validation batches found")
        
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
        self.class_names = class_names or ["background", "crack"]
        
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
            # SSD models expect a list of tensors
            if len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1:
                # Already batched, but need to convert to list
                predictions = self.model([input_tensor[0]])
            else:
                # Add to list for model
                predictions = self.model([input_tensor])
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


# ==================== DATASET WRAPPER FOR TRANSFORMATIONS ====================

class TransformedDataset(Dataset):
    """Dataset wrapper that applies transformations to another dataset"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        
        # Handle boxes - ensure they're in the right format
        boxes = target["boxes"].numpy() if isinstance(target["boxes"], torch.Tensor) else target["boxes"]
        labels = target["labels"].numpy() if isinstance(target["labels"], torch.Tensor) else target["labels"]
        
        # Check if we have bounding boxes
        if len(boxes) > 0:
            # Apply transformations with bounding boxes
            try:
                transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)
                img = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']
            except Exception as e:
                print(f"Error applying transform with bboxes: {e}")
                # Fallback to just image transform
                transformed = A.Compose([
                    A.Resize(320, 320),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])(image=img)
                img = transformed['image']
                boxes = []
                labels = []
        else:
            # Apply only image transformations
            transform_img_only = A.Compose([
                A.Resize(320, 320),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            transformed = transform_img_only(image=img)
            img = transformed['image']
        
        # Convert back to tensor format
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros(0, dtype=torch.int64)
        
        return img, target


# ==================== MAIN TRAINING & DEPLOYMENT PIPELINE ====================

def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    This function needs to be outside of any other function to be picklable.
    """
    return tuple(zip(*batch))

def main():
    # Define paths with proper handling for directory names
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # List available directories to help identify where images might be
    print("Available directories and files:")
    for item in os.listdir(current_dir):
        if os.path.isdir(os.path.join(current_dir, item)):
            print(f"  Directory: {item}")
        else:
            print(f"  File: {item}")
    
    # Use the correct directory names that we found
    images_dir = os.path.join(current_dir, "croped images")
    labels_dir = os.path.join(current_dir, "labels")
    
    # Check what files are in the images directory
    print(f"\nChecking contents of images directory: {images_dir}")
    if os.path.exists(images_dir):
        # Check for subdirectories
        subdirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
        if subdirs:
            print(f"Found {len(subdirs)} subdirectories: {', '.join(subdirs)}")
            
            # Check for image files in each subdirectory
            total_images = 0
            for subdir in subdirs:
                subdir_path = os.path.join(images_dir, subdir)
                files = os.listdir(subdir_path)
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                print(f"  {subdir}: Found {len(image_files)} image files out of {len(files)} total files")
                
                # Show some example files
                if files:
                    print(f"  Example files in {subdir} (first 5):")
                    for f in files[:5]:
                        print(f"    {f}")
                
                total_images += len(image_files)
            
            print(f"Total image files across all subdirectories: {total_images}")
            
            if total_images == 0:
                print("Warning: No image files found in subdirectories!")
                # Check what file extensions exist
                all_files = []
                for subdir in subdirs:
                    subdir_path = os.path.join(images_dir, subdir)
                    all_files.extend(os.listdir(subdir_path))
                
                extensions = {}
                for f in all_files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in extensions:
                        extensions[ext] += 1
                    else:
                        extensions[ext] = 1
                
                print("File extensions found:")
                for ext, count in extensions.items():
                    print(f"  {ext}: {count} files")
    
    # Set up training and validation directories
    train_img_dir = images_dir
    train_annot_dir = labels_dir
    val_img_dir = images_dir
    val_annot_dir = labels_dir
    
    # Check if directories exist
    for dir_path, dir_name in [(train_img_dir, "Images"), (train_annot_dir, "Labels")]:
        if not os.path.exists(dir_path):
            print(f"Error: {dir_name} directory '{dir_path}' does not exist.")
            print("Please make sure the directories exist or update the paths in the code.")
            return  # Exit the function without proceeding further
    
    # Print the number of annotation files found
    annot_files = [f for f in os.listdir(labels_dir) if f.endswith('.xml')]
    print(f"Found {len(annot_files)} annotation files in {labels_dir}")
    
    if len(annot_files) == 0:
        print("Error: No annotation files found. Cannot proceed with training.")
        return
    
    print(f"Using images directory: {images_dir}")
    print(f"Using labels directory: {labels_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define class names based on subdirectories
    class_names = ["background"] + subdirs if subdirs else ["background", "crack"]
    print(f"Using class names: {class_names}")
    num_classes = len(class_names)
    
    # Create dataset - we'll use a single dataset and split it
    full_dataset = DetectionDataset(
        img_dir=images_dir,
        annot_dir=labels_dir,
        transform=None,  # No transform here - will apply later
        class_names=class_names
    )
    
    # Print dataset stats
    print(f"Full dataset size: {len(full_dataset)}")
    
    # Split the dataset into training and validation
    dataset_size = len(full_dataset)
    val_size = int(0.2 * dataset_size)  # 20% for validation
    train_size = dataset_size - val_size
    
    # Use random_split to create the splits
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Apply transformations
    train_dataset = TransformedDataset(train_dataset, get_transform(train=True))
    val_dataset = TransformedDataset(val_dataset, get_transform(train=False))
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Disable multiprocessing for reliability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # Disable multiprocessing for reliability
    )
    
    # Create calibration loader for quantization (subset of validation data)
    calibration_dataset = torch.utils.data.Subset(val_dataset, indices=range(min(50, len(val_dataset))))
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # Disable multiprocessing for reliability
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
        num_epochs=100  # Standard number of epochs
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

def raspi_deployment(headless=False, source_type="camera", source_path=None):
    """
    Script to be run on Raspberry Pi for live inference
    
    Args:
        headless: If True, run in headless mode without trying to display frames
        source_type: Type of input source - "camera", "video", or "image"
        source_path: Path to the image or video file if source_type is "image" or "video"
    """
    # Check if OpenCV has GUI support
    has_gui = False
    try:
        # Try to create a window to check if GUI is supported
        cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.destroyAllWindows()
        has_gui = True
    except Exception as e:
        print(f"OpenCV GUI not available: {e}")
        print("Running in headless mode (no display)")
        headless = True
    
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or create the model
    model_path = "mobilenet_ssd_trained.pth"  # Use the non-quantized model
    
    class_names = ["background", "degradable", "non degradable"]
    num_classes = len(class_names)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = create_mobilenet_ssd(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model file {model_path} not found. Creating a new model.")
        model = create_mobilenet_ssd(num_classes=num_classes)
    
    model = model.to(device)
    model.eval()
    print("Model ready")
    
    # Create transform for preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Function to run detection
    def detect_objects(image, confidence_threshold=0.5):
        # Convert image to PIL if it's numpy array
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
            
        # Get original dimensions
        orig_width, orig_height = image_pil.size
        
        # Preprocess the image
        input_tensor = preprocess(image_pil)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            predictions = model(input_tensor)
        inference_time = time.time() - start_time
        
        # Process predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores >= confidence_threshold
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
        
        # Build result list
        results = []
        for box, score, label in zip(scaled_boxes, scores, labels):
            class_name = class_names[label]
            results.append({
                'box': box,
                'score': float(score),
                'class_name': class_name,
                'class_id': int(label)
            })
        
        return results, inference_time
    
    # Handle different source types
    if source_type == "image":
        # Process a single image
        if source_path is None or not os.path.exists(source_path):
            print("Error: Image file not found or not specified")
            return
            
        print(f"Processing image: {source_path}")
        img = cv2.imread(source_path)
        if img is None:
            print(f"Error: Could not read image from {source_path}")
            return
            
        # Run detection
        start_time = time.time()
        detections, inference_time = detect_objects(img)
        total_time = time.time() - start_time
        
        # Print results
        print(f"Processed image in {total_time:.2f} seconds (inference: {inference_time:.2f} seconds)")
        if len(detections) > 0:
            print(f"Found {len(detections)} detections:")
            for i, det in enumerate(detections):
                print(f"  {i+1}: {det['class_name']} ({det['score']:.2f}) at {det['box']}")
        else:
            print("No detections found")
            
        # Draw bounding boxes
        for detection in detections:
            box = detection['box']
            score = detection['score']
            class_name = detection['class_name']
            
            xmin, ymin, xmax, ymax = box
            
            # Draw bounding box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            cv2.putText(img, label, (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Display results if GUI is available
        if not headless and has_gui:
            try:
                cv2.imshow("Detection Result", img)
                print("Press any key to exit...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error displaying image: {e}")
                
        # Save the output image
        output_path = f"output_{os.path.basename(source_path)}"
        cv2.imwrite(output_path, img)
        print(f"Result saved to {output_path}")
        
    elif source_type == "video":
        # Process a video file
        if source_path is None or not os.path.exists(source_path):
            print("Error: Video file not found or not specified")
            return
            
        print(f"Processing video: {source_path}")
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {source_path}")
            return
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Create output video writer
        output_path = f"output_{os.path.basename(source_path)}"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        frame_count = 0
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Processing frame {frame_count}/{total_frames}...")
                
                # Run detection
                detections, inference_time = detect_objects(frame)
                
                # Draw bounding boxes
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
                
                # Write frame to output video
                out.write(frame)
                
                # Display frame if GUI is available
                if not headless and has_gui and frame_count % 5 == 0:  # Only show every 5th frame
                    try:
                        cv2.imshow("Detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        print(f"Error displaying frame: {e}")
                        headless = True
        
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        finally:
            # Calculate processing speed
            total_time = time.time() - start_time
            processing_fps = frame_count / total_time if total_time > 0 else 0
            
            # Release resources
            cap.release()
            out.release()
            if has_gui:
                cv2.destroyAllWindows()
                
            print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({processing_fps:.2f} FPS)")
            print(f"Result saved to {output_path}")
            
    else:  # Camera input
        # Try different camera sources
        sources = []
        
        # If source_path is provided and is a URL
        if source_path and (source_path.startswith("http://") or source_path.startswith("rtsp://")):
            sources.append(("Custom URL", source_path))
            
        # Add default IP camera options
        ip = "192.168.29.131"  # Replace with your phone's IP address
        sources.append(("DroidCam", f"http://{ip}:4747/video"))
        sources.append(("IP Webcam", f"http://{ip}:8080/video"))
        
        # Add local camera options
        for i in range(3):  # Try first 3 camera indices
            sources.append((f"Local Camera {i}", i))
        
        # Try each source
        cap = None
        
        for source_name, source in sources:
            print(f"Trying {source_name}: {source}")
            try:
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret:
                        print(f"Successfully connected to {source_name}")
                        break
                    else:
                        print(f"Connected to {source_name} but could not read frame")
                        cap.release()
                        cap = None
                else:
                    print(f"Failed to connect to {source_name}")
                    cap = None
            except Exception as e:
                print(f"Error connecting to {source_name}: {e}")
                cap = None
        
        if cap is None:
            print("Could not connect to any camera source. Exiting.")
            return
        
        # For FPS calculation
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        print("Starting inference. Press Ctrl+C to exit.")
        
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to capture image")
                    # Wait a bit and retry
                    time.sleep(1)
                    continue
                
                # Run detection
                detections, inference_time = detect_objects(frame)
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                elapsed = current_time - start_time
                
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = current_time
                    
                    # Print detection results
                    if len(detections) > 0:
                        print(f"Detected {len(detections)} objects:")
                        for i, det in enumerate(detections):
                            print(f"  {i+1}: {det['class_name']} ({det['score']:.2f}) at {det['box']}")
                    else:
                        print("No detections")
                        
                    print(f"FPS: {fps:.1f}, Inference time: {inference_time*1000:.1f}ms")
                
                # Draw results on frame if GUI is available
                if not headless and has_gui:
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
                    
                    try:
                        # Display the frame
                        cv2.imshow('Object Detection', frame)
                        
                        # Press 'q' to exit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        print(f"Error displaying frame: {e}")
                        print("Switching to headless mode...")
                        headless = True
                        has_gui = False
                
                # If in headless mode, add a slight delay to avoid consuming too much CPU
                if headless:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error during inference: {e}")
        finally:
            # Release resources
            cap.release()
            if has_gui:
                cv2.destroyAllWindows()
            print("Inference stopped, resources released")


if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn' for Windows compatibility
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="MobileNet SSD training and deployment")
    parser.add_argument("mode", choices=["train", "deploy"], default="train", nargs="?",
                       help="Mode to run: 'train' to train the model, 'deploy' for inference")
    parser.add_argument("--source", choices=["camera", "image", "video"], default="camera",
                       help="Source type for deployment mode")
    parser.add_argument("--path", type=str, default=None,
                       help="Path to image or video file for deployment mode")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode without GUI")
    
    args = parser.parse_args()
    
    if args.mode == "deploy":
        # For deployment
        print(f"Running in deployment mode with source_type={args.source}")
        raspi_deployment(
            headless=args.headless,
            source_type=args.source,
            source_path=args.path
        )
    else:
        # For training
        print("Running in training mode")
        main()
    
    pass
