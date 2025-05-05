#!/usr/bin/env python3
"""
Military Obstacle Detection System (YOLOv8-Based)

This script detects military obstacles in aerial images and outputs their
real-world positions and risk scores for path planning.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Military obstacle detection from aerial images")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLOv8 weights file")
    parser.add_argument("--source", type=str, required=True, help="Path to source images (glob pattern)")
    parser.add_argument("--risk-config", type=str, required=True, help="Path to risk configuration YAML")
    parser.add_argument("--save", action="store_true", help="Save annotated images")
    parser.add_argument("--drone-altitude", type=float, default=50.0, help="Drone altitude in meters")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="Scale factor for pixel-to-meter conversion")
    return parser.parse_args()


def load_risk_config(config_path):
    """Load risk configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["risk_weights"]


def pixel_to_meter(px_coords, image_height, drone_altitude, scale_factor):
    """Convert pixel coordinates to real-world meters."""
    meters_per_pixel = scale_factor * drone_altitude / image_height
    return px_coords * meters_per_pixel


def process_image(image_path, model, risk_weights, drone_altitude, scale_factor, save_vis=False):
    """Process a single image and detect obstacles."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return []
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Run YOLOv8 detection
    results = model(img, verbose=False)
    
    # Process detections
    detections = []
    
    for r in results:
        boxes = r.boxes
        
        # Process each detection
        for box in boxes:
            # Get class, confidence and bounding box
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            confidence = box.conf[0].item()
            
            # Get bounding box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Calculate center of bounding box in pixels
            center_x_px = (x1 + x2) / 2
            center_y_px = (y1 + y2) / 2
            
            # Convert to real-world coordinates (assuming top-left is origin)
            x_meters = pixel_to_meter(center_x_px, height, drone_altitude, scale_factor)
            y_meters = pixel_to_meter(center_y_px, height, drone_altitude, scale_factor)
            
            # Get risk value from config (default 0.3 if not found)
            risk = risk_weights.get(cls_name, 0.3)
            
            # Add to detections list
            detections.append({
                "class": cls_name,
                "x": round(x_meters, 2),
                "y": round(y_meters, 2),
                "risk": risk,
                "confidence": round(confidence, 2)
            })
            
            # Draw on image if save_vis is True
            if save_vis:
                # Draw bounding box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label
                label = f"{cls_name}: {confidence:.2f}, Risk: {risk}"
                cv2.putText(img, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save annotated image if requested
    if save_vis:
        output_img_path = str(Path(image_path).with_suffix('.det.jpg'))
        cv2.imwrite(output_img_path, img)
        print(f"Saved annotated image to {output_img_path}")
    
    return detections


def main():
    """Main function."""
    args = parse_args()
    
    # Load YOLOv8 model
    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)
    
    # Load risk configuration
    print(f"Loading risk configuration from {args.risk_config}...")
    risk_weights = load_risk_config(args.risk_config)
    
    # Get list of images
    image_paths = glob.glob(args.source)
    if not image_paths:
        print(f"No images found matching pattern: {args.source}")
        return
    
    print(f"Processing {len(image_paths)} images...")
    
    # Process each image
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        
        # Detect obstacles
        detections = process_image(
            image_path, 
            model, 
            risk_weights, 
            args.drone_altitude, 
            args.scale_factor,
            args.save
        )
        
        # Save detections to JSON
        output_json_path = str(Path(image_path).with_suffix('.detections.json'))
        with open(output_json_path, 'w') as f:
            json.dump(detections, f, indent=2)
        
        print(f"Saved {len(detections)} detections to {output_json_path}")
    
    print("Done!")


if __name__ == "__main__":
    main() 