import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom

def yolo_to_pascal_voc(img_path, yolo_path, xml_path, class_names):
    # Read image to get dimensions
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    
    # Read YOLO annotation
    with open(yolo_path, 'r') as f:
        lines = f.readlines()
    
    # Create XML structure
    annotation = ET.Element('annotation')
    
    # Add image info
    ET.SubElement(annotation, 'filename').text = os.path.basename(img_path)
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img_width)
    ET.SubElement(size, 'height').text = str(img_height)
    ET.SubElement(size, 'depth').text = '3'
    
    # Process each object
    for line in lines:
        if line.strip():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            class_id = int(class_id)
            
            # Convert YOLO coordinates to Pascal VOC
            x_min = int((x_center - width/2) * img_width)
            y_min = int((y_center - height/2) * img_height)
            x_max = int((x_center + width/2) * img_width)
            y_max = int((y_center + height/2) * img_height)
            
            # Ensure coordinates are within image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)
            
            # Add object to XML
            obj = ET.SubElement(annotation, 'object')
            ET.SubElement(obj, 'name').text = class_names[class_id]
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            
            bbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bbox, 'xmin').text = str(x_min)
            ET.SubElement(bbox, 'ymin').text = str(y_min)
            ET.SubElement(bbox, 'xmax').text = str(x_max)
            ET.SubElement(bbox, 'ymax').text = str(y_max)
    
    # Write XML file
    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="    ")
    with open(xml_path, 'w') as f:
        f.write(xml_str)

# Example usage
def convert_dataset(img_dir, yolo_dir, xml_dir, class_names):
    os.makedirs(xml_dir, exist_ok=True)
    
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(img_dir, filename)
            yolo_path = os.path.join(yolo_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            xml_path = os.path.join(xml_dir, filename.replace('.jpg', '.xml').replace('.png', '.xml'))
            
            if os.path.exists(yolo_path):
                yolo_to_pascal_voc(img_path, yolo_path, xml_path, class_names)
