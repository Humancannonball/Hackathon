#!/usr/bin/env python3
"""
Convert DOTA dataset annotations to YOLOv8 format.
DOTA uses text files with rotated bounding box annotations.
YOLOv8 uses normalized coordinates (x_center, y_center, width, height).
"""

import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
import cv2

# Define the classes we want to keep from DOTA
# Focus on military-relevant classes
MILITARY_CLASSES = {
    'helicopter': 0,
    'airplane': 1,  # can be military aircraft
    'ship': 2,      # can be military vessel
    'vehicle': 3,   # can be military vehicle/tank
    'storage-tank': 4,
    'bridge': 5,
    'harbor': 6
    # Add more classes as needed
}

def parse_dota_annotation(ann_file):
    """Parse DOTA annotation file and return object annotations."""
    objects = []
    with open(ann_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
                
            # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
            # Extracts the 8 coordinates for rotated box
            coords = [float(p) for p in parts[:8]]
            class_name = parts[8]
            
            # Only keep military-relevant classes
            if class_name.lower() in MILITARY_CLASSES:
                objects.append({
                    'coords': coords,
                    'class': class_name.lower()
                })
    
    return objects

def rotated_box_to_yolo(coords, img_width, img_height):
    """Convert rotated box coordinates to YOLO format."""
    # Create a numpy array of the coordinates
    coords = np.array(coords).reshape(4, 2)
    
    # Get the center of the rotated box
    center_x = np.mean(coords[:, 0]) / img_width
    center_y = np.mean(coords[:, 1]) / img_height
    
    # Calculate width and height (using maximum distances)
    width = np.max(np.linalg.norm(coords[0] - coords[1], coords[2] - coords[3])) / img_width
    height = np.max(np.linalg.norm(coords[1] - coords[2], coords[3] - coords[0])) / img_height
    
    # Ensure values are within [0, 1]
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return center_x, center_y, width, height

def convert_dataset(image_dir, annotation_dir, output_dir, split=(0.8, 0.1, 0.1)):
    """Convert DOTA dataset to YOLOv8 format and split into train/val/test."""
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)
    
    # Get image files
    image_files = glob.glob(os.path.join(image_dir, '*.png')) + \
                 glob.glob(os.path.join(image_dir, '*.jpg')) + \
                 glob.glob(os.path.join(image_dir, '*.tif'))
    
    # Split dataset
    train_files, test_val_files = train_test_split(image_files, test_size=(split[1]+split[2]), random_state=42)
    val_files, test_files = train_test_split(test_val_files, test_size=split[2]/(split[1]+split[2]), random_state=42)
    
    # Process each split
    process_split(train_files, annotation_dir, output_dir, 'train')
    process_split(val_files, annotation_dir, output_dir, 'val')
    process_split(test_files, annotation_dir, output_dir, 'test')
    
    # Create dataset.yaml
    create_dataset_yaml(output_dir)
    
    print(f"Converted {len(image_files)} images to YOLOv8 format")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

def process_split(image_files, annotation_dir, output_dir, split_name):
    """Process a dataset split."""
    for img_path in image_files:
        # Get base filename without extension
        basename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Find corresponding annotation file
        ann_path = os.path.join(annotation_dir, f"{basename}.txt")
        if not os.path.exists(ann_path):
            print(f"Annotation not found for {basename}, skipping")
            continue
        
        # Get image dimensions
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Parse annotations
        objects = parse_dota_annotation(ann_path)
        
        # Skip if no military objects
        if not objects:
            continue
        
        # Copy image to output directory
        dst_img_path = os.path.join(output_dir, 'images', split_name, os.path.basename(img_path))
        shutil.copy(img_path, dst_img_path)
        
        # Create YOLO annotation file
        yolo_path = os.path.join(output_dir, 'labels', split_name, f"{basename}.txt")
        with open(yolo_path, 'w') as f:
            for obj in objects:
                # Convert to YOLO format
                class_id = MILITARY_CLASSES[obj['class']]
                x_center, y_center, width, height = rotated_box_to_yolo(obj['coords'], img_width, img_height)
                
                # Write to file
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def create_dataset_yaml(output_dir):
    """Create dataset.yaml file for YOLOv8."""
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        
        f.write(f"nc: {len(MILITARY_CLASSES)}\n")
        f.write("names:\n")
        
        # Sort classes by ID
        classes = sorted(MILITARY_CLASSES.items(), key=lambda x: x[1])
        for name, _ in classes:
            f.write(f"  {_}: {name}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert DOTA dataset to YOLOv8 format")
    parser.add_argument('--image-dir', required=True, help='Directory containing DOTA images')
    parser.add_argument('--annotation-dir', required=True, help='Directory containing DOTA annotations')
    parser.add_argument('--output-dir', required=True, help='Output directory for YOLOv8 dataset')
    args = parser.parse_args()
    
    convert_dataset(args.image_dir, args.annotation_dir, args.output_dir)

if __name__ == "__main__":
    main() 