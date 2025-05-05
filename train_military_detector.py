#!/usr/bin/env python3
"""
Train a YOLOv8 model for military object detection using the preprocessed DOTA dataset.
"""

import argparse
import os
import yaml
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for military object detection')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset.yaml file')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size: n (nano), s (small), m (medium), l (large), x (xlarge)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--device', type=str, default='', help='Training device (empty for auto, or cuda:0, cpu, etc.)')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads')
    parser.add_argument('--name', type=str, default='military_detector', help='Project name')
    parser.add_argument('--exist-ok', action='store_true', help='Overwrite existing project')
    
    return parser.parse_args()

def validate_dataset(data_yaml):
    """Validate dataset configuration."""
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")
    
    with open(data_yaml, 'r') as f:
        try:
            data = yaml.safe_load(f)
            required_keys = ['path', 'train', 'val', 'nc', 'names']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key in dataset.yaml: {key}")
            
            # Verify paths exist
            base_path = data['path']
            train_path = os.path.join(base_path, data['train'])
            val_path = os.path.join(base_path, data['val'])
            
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Training data not found: {train_path}")
            if not os.path.exists(val_path):
                raise FileNotFoundError(f"Validation data not found: {val_path}")
                
            print(f"Dataset validated: {len(data['names'])} classes")
            print(f"Classes: {', '.join(data['names'].values())}")
            
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing dataset.yaml: {e}")

def train_model(args):
    """Train the YOLOv8 model."""
    # Validate dataset first
    validate_dataset(args.data)
    
    # Initialize model
    model_name = f"yolov8{args.model_size}.pt"
    print(f"Initializing model: {model_name}")
    model = YOLO(model_name)
    
    # Set training arguments
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.img_size,
        'batch': args.batch_size,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'workers': args.workers,
        'pretrained': True,
        'optimizer': 'Adam',  # Using Adam optimizer
        'patience': 50,       # Early stopping patience
        'save': True,         # Save checkpoints
        'save_period': 10,    # Save checkpoint every 10 epochs
    }
    
    # Add device if specified
    if args.device:
        train_args['device'] = args.device
    
    # Resume training if requested
    if args.resume:
        train_args['resume'] = True
    
    # Print training configuration
    print("Training configuration:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")
    
    # Start training
    print("Starting training...")
    results = model.train(**train_args)
    
    return results

def main():
    args = parse_args()
    try:
        results = train_model(args)
        print(f"Training completed! Best weights saved at: {results.best}")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main() 