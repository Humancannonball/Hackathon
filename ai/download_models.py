#!/usr/bin/env python3
"""
Download pretrained YOLO OBB models for oriented object detection
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description='Download pretrained YOLO OBB models')
    parser.add_argument('--all', action='store_true', help='Download all models')
    parser.add_argument('--nano', action='store_true', help='Download nano model (fastest)')
    parser.add_argument('--small', action='store_true', help='Download small model')
    parser.add_argument('--medium', action='store_true', help='Download medium model')
    parser.add_argument('--large', action='store_true', help='Download large model')
    parser.add_argument('--xlarge', action='store_true', help='Download extra large model (best quality)')
    return parser.parse_args()

def check_gpu():
    """Check if GPU is available and print info"""
    print("\n=== GPU Information ===")
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s).")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            print(f"  GPU {i}: {device_name} - Total Memory: {memory_total:.2f} GB")
        
        # Recommend models based on GPU memory
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print("\n=== Model Recommendations ===")
        if memory < 4:
            print("  Limited GPU memory (<4GB): Use nano model (yolo11n-obb.pt)")
        elif memory < 8:
            print("  Moderate GPU memory (4-8GB): Use nano or small model (yolo11n/s-obb.pt)")
        elif memory < 12:
            print("  Good GPU memory (8-12GB): Use small or medium model (yolo11s/m-obb.pt)")
        elif memory < 16:
            print("  High GPU memory (12-16GB): Use medium or large model (yolo11m/l-obb.pt)")
        else:
            print("  Excellent GPU memory (>16GB): Use any model, including extra large (yolo11x-obb.pt)")
    else:
        print("No GPU detected. Using CPU will be slow for inference.")
        print("  Recommended model for CPU: yolo11n-obb.pt (nano)")
    print("=========================")

def download_model(model_name):
    """Download a specific model and provide information"""
    print(f"\n=== Downloading {model_name} ===")
    try:
        # Create a YOLO model from the pretrained weights
        model = YOLO(model_name)
        
        # Get the model file path
        model_path = Path(model_name).resolve()
        if model_path.exists():
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✓ Model {model_name} downloaded successfully")
            print(f"  - Size: {model_size_mb:.2f} MB")
            print(f"  - Path: {model_path}")
        else:
            print(f"✗ Something went wrong. Model file not found at {model_path}")
        
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")

def main():
    args = parse_arguments()
    
    # Check GPU and recommend models
    check_gpu()
    
    models_to_download = []
    
    # Determine which models to download
    if args.all:
        models_to_download = ['yolo11n-obb.pt', 'yolo11s-obb.pt', 'yolo11m-obb.pt', 'yolo11l-obb.pt', 'yolo11x-obb.pt']
    else:
        if args.nano:
            models_to_download.append('yolo11n-obb.pt')
        if args.small:
            models_to_download.append('yolo11s-obb.pt')
        if args.medium:
            models_to_download.append('yolo11m-obb.pt')
        if args.large:
            models_to_download.append('yolo11l-obb.pt')
        if args.xlarge:
            models_to_download.append('yolo11x-obb.pt')
        
        # Default to medium model if nothing specified
        if not models_to_download:
            print("No specific model selected, defaulting to medium model (yolo11m-obb.pt)")
            models_to_download = ['yolo11m-obb.pt']
    
    # Download the selected models
    print(f"\nDownloading {len(models_to_download)} model(s)...\n")
    for model_name in models_to_download:
        download_model(model_name)
    
    print("\n=== All downloads complete ===")
    print("To use these models with video.py:")
    print("  python video.py --model [MODEL_NAME] --source v.mp4")
    print("  Example: python video.py --model yolo11m-obb.pt --source v.mp4")

if __name__ == "__main__":
    main()
