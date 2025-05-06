#!/usr/bin/env python3
"""
Video Processing with YOLOv8 OBB - A simple script to process videos with oriented object detection
"""

import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process video with YOLO11n OBB model')
    parser.add_argument(
        '--model', type=str,
        default='yolo11m-obb.pt',  # Upgraded default to medium model for better detection
        help='Path to YOLOv8 OBB model (default: yolo11m-obb.pt). Options: yolo11n/s/m/l/x-obb.pt'
    )
    parser.add_argument(
        '--source', type=str,
        default='v.mp4',
        help='Source video file path (default: v.mp4)'
    )
    parser.add_argument(
        '--output', type=str,
        default=None,
        help='Output video file path (default: auto-generated based on input)'
    )
    parser.add_argument(
        '--conf', type=float,
        default=0.2,  # Lowered confidence threshold slightly to detect more objects
        help='Confidence threshold (default: 0.2)'
    )
    parser.add_argument(
        '--iou', type=float,
        default=0.7,
        help='NMS IoU threshold (default: 0.7)'
    )
    parser.add_argument(
        '--img-size', type=int,
        default=1024,  # Increased default image size for better detection
        help='Image size for inference (default: 1024)'
    )
    parser.add_argument(
        '--device', type=str,
        default='',
        help='Device to run on (e.g., cuda:0 or cpu)'
    )
    parser.add_argument(
        '--view', action='store_true',
        help='Display video during processing'
    )
    parser.add_argument(
        '--info', action='store_true',
        help='Display FPS and object count on output video'
    )
    parser.add_argument(
        '--nosave', action='store_true',
        help="Don't save the output video"
    )
    parser.add_argument(
        '--classes', type=int, nargs='+',
        help='Filter by class, e.g. --classes 0 9 10 for planes and vehicles'
    )
    # Add new arguments for improved detection
    parser.add_argument(
        '--half', action='store_true',
        help='Use FP16 half-precision inference (faster on compatible GPUs)'
    )
    parser.add_argument(
        '--augment', action='store_true',
        help='Apply test-time augmentation for better detection'
    )
    parser.add_argument(
        '--tracker', type=str, 
        choices=['botsort.yaml', 'bytetrack.yaml', 'none'], 
        default='bytetrack.yaml',  # Fixed to match YOLO's expected format
        help='Object tracker to use (default: bytetrack.yaml)'
    )
    parser.add_argument(
        '--persist', action='store_true',
        help='Enable detection persistence to reduce flickering'
    )
    parser.add_argument(
        '--agnostic', action='store_true',
        help='Class-agnostic NMS for better detection'
    )
    parser.add_argument(
        '--export-coords', type=str,
        default=None,
        help='Export object coordinates to specified JSON file for path planning'
    )
    
    return parser.parse_args()


def select_device(device_preference=''):
    """Select the appropriate processing device"""
    if torch.cuda.is_available() and device_preference != 'cpu':
        if device_preference:
            # Use specified GPU if available
            device = f'cuda:{device_preference}'
        else:
            # Default to first GPU
            device = 'cuda:0'
        
        gpu_id = int(device.split(':')[1]) if ':' in device else 0
        device_name = torch.cuda.get_device_name(gpu_id)
        memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)
        print(f"Using GPU: {device_name} ({memory_total:.2f} GB)")
    else:
        device = 'cpu'
        print("Using CPU for inference (may be slower)")
    
    return device


def process_video(args):
    """Process video with the YOLOv8 OBB model"""
    # Set environment variable to prevent memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Select device
    device = select_device(args.device)
    
    # Generate output path if not specified
    if args.output is None and not args.nosave:
        source_path = Path(args.source)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if this is an OBB model
        is_obb_model = 'obb' in args.model.lower()
        
        # Choose appropriate output directory based on model type
        if is_obb_model:
            output_dir = Path('runs/obb/predict')
        else:
            output_dir = Path('runs/detect/predict')
            
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the next available numbered directory
        existing_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        next_num = 1 if not existing_dirs else max(int(d.name) for d in existing_dirs) + 1
        
        # Create the output path using the same pattern as the model's save_dir
        predict_dir = output_dir / str(next_num)
        args.output = str(predict_dir / f"{source_path.stem}.mp4")
    
    # Model suggestions for better performance
    model_suggestions = {
        'yolo11n-obb.pt': 'Fast but less accurate',
        'yolo11s-obb.pt': 'Good balance of speed and accuracy',
        'yolo11m-obb.pt': 'Better accuracy, moderate speed',
        'yolo11l-obb.pt': 'High accuracy, slower',
        'yolo11x-obb.pt': 'Highest accuracy, slowest'
    }
    
    # Print model recommendations
    print("\n=== Model Selection Guide ===")
    print(f"Current model: {args.model}")
    for model, description in model_suggestions.items():
        if model == args.model:
            print(f"-> {model}: {description} (CURRENT)")
        else:
            print(f"   {model}: {description}")
    print("===========================\n")
    
    # Load model
    print(f"Loading model {args.model}...")
    try:
        model = YOLO(args.model)
        # Check if this is an OBB model
        is_obb_model = hasattr(model, 'task') and model.task == 'obb'
        model_type = "OBB" if is_obb_model else "Standard"
        print(f"Loaded {model_type} YOLO model successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Verify source exists
    if not os.path.isfile(args.source):
        print(f"Error: Source file '{args.source}' not found")
        return
    
    # Process the video
    print(f"Processing video: {args.source}")
    if not args.nosave:
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {args.output}")
    else:
        print("Output will not be saved (--nosave flag used)")
    
    try:
        start_time = time.time()
        
        # Enable visualization of detections with labels
        visual_settings = {
            'line_width': 2,
            'show_boxes': True,
            'show_labels': True,
            'show_conf': True,
            'vid_stride': 1       # Process every frame
        }
        
        # Configure tracker settings - fixed to match YOLO's expected format
        if args.tracker != 'none':
            visual_settings['tracker'] = args.tracker
        
        # Dictionary to store coordinates for path planning
        # Keys will be frame numbers (as strings for JSON compatibility),
        # values will be dictionaries containing frame number and detected objects.
        object_coords = {}
        
        # Run prediction with enhanced parameters
        results = model.predict(
            source=args.source,
            save=not args.nosave,  # Save output video if not nosave
            save_txt=False,        # Don't save text results by default
            # avoid double directory: use parent of output-dir as project, and last folder name as name
            project=None if not args.output else os.path.dirname(os.path.dirname(args.output)),
            name=None if not args.output else os.path.basename(os.path.dirname(args.output)),
            exist_ok=True,         # Overwrite existing output
            stream=True,           # Enable streaming for better memory usage
            imgsz=args.img_size,   # Set image size for processing
            conf=args.conf,        # Confidence threshold
            iou=args.iou,          # NMS IoU threshold
            device=device,         # Use selected device
            show=args.view,        # Show video while processing if requested
            classes=args.classes,  # Filter by classes if specified
            verbose=False,         # Reduce verbose output
            half=args.half,        # FP16 inference if requested
            augment=args.augment,  # Use test-time augmentation if requested
            agnostic_nms=args.agnostic,  # Class-agnostic NMS
            **visual_settings      # Apply visualization settings
        )
        
        # Determine final output path - check first result for save_dir
        save_dir = None
        final_output_path = None
        frame_count = 0
        total_objects = 0
        # Process results and extract coordinates for path planning
        for i, result in enumerate(results):
            # Only check the first result for save_dir to avoid repeated checks
            if i == 0 and hasattr(result, 'save_dir'):
                save_dir = result.save_dir
                # Always detect the actual saved video (mp4 or avi)
                save_dir_path = Path(save_dir)
                # Look for common video file extensions
                video_files = [f for f in save_dir_path.iterdir() if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
                if video_files:
                    # Prefer mp4 if multiple exist, otherwise take the first found
                    mp4_files = [f for f in video_files if f.suffix.lower() == '.mp4']
                    final_output_path = str(mp4_files[0]) if mp4_files else str(video_files[0])

            frame_count += 1
            frame_objects = []
            
            # On first frame, store the save directory if available
            if frame_count == 1 and hasattr(result, 'save_dir'):
                save_dir = result.save_dir
            
            # Extract coordinates from detections
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                # Filter boxes by confidence and class (to match what would be drawn)
                filtered_boxes = []
                for box in boxes:
                    conf = float(box.conf.item()) if hasattr(box, 'conf') else 0.0
                    cls = int(box.cls.item()) if hasattr(box, 'cls') else -1
                    # Filter by confidence threshold
                    if conf < args.conf:
                        continue
                    # Filter by class if --classes is set
                    if args.classes is not None and cls not in args.classes:
                        continue
                    filtered_boxes.append(box)
                current_frame_object_count = len(filtered_boxes)
                total_objects += current_frame_object_count

                # Process each detection (only those that would be drawn)
                for box_idx, box in enumerate(filtered_boxes):
                    # Get basic info
                    box_id = int(box.id.item()) if hasattr(box, 'id') and box.id is not None else box_idx
                    cls = int(box.cls.item()) if hasattr(box, 'cls') else -1
                    conf = float(box.conf.item()) if hasattr(box, 'conf') else 0.0
                    
                    # Get coordinates
                    if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > box_idx:
                        coords = result.obb.xyxyxyxy[box_idx].cpu().numpy().tolist()
                        obj_type = "obb"
                    elif hasattr(box, 'xyxy') and box.xyxy is not None and len(box.xyxy) > 0:
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        obj_type = "bbox"
                    else:
                        continue

                    # Calculate center point
                    if obj_type == "obb":
                        if isinstance(coords, list) and len(coords) == 8:
                            x_coords = coords[0::2]
                            y_coords = coords[1::2]
                            center_x = sum(x_coords) / len(x_coords)
                            center_y = sum(y_coords) / len(y_coords)
                        else:
                             center_x, center_y = -1, -1
                    elif obj_type == "bbox":
                        if isinstance(coords, list) and len(coords) == 4:
                            x1, y1, x2, y2 = coords
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                        else:
                            center_x, center_y = -1, -1
                    else:
                         center_x, center_y = -1, -1

                    frame_objects.append({
                        "id": box_id,
                        "class": cls,
                        "confidence": conf,
                        "center": [center_x, center_y],
                        "coordinates": coords,
                        "type": obj_type
                    })
            
            # Store coordinates for this frame using frame_count as the key
            # The value includes the frame number and the list of objects
            if frame_objects:
                # Use string key for JSON compatibility, store frame number inside value
                object_coords[str(frame_count)] = {
                    "frame_number": frame_count,
                    "objects": frame_objects
                }
            
            # Status update every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count} frames ({fps:.2f} FPS)")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        # Print summary
        print("\n===== Processing Complete =====")
        print(f"Video: {args.source}")
        print(f"Frames processed: {frame_count}")
        print(f"Objects detected: {total_objects}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Total time: {total_time:.2f} seconds")
        
        # Export coordinates for path planning if requested
        if args.export_coords and object_coords:
            try:
                import json
                output_file = args.export_coords
                # if no directory specified, save JSON next to the video output
                if not os.path.dirname(output_file) and save_dir:
                    output_file = os.path.join(save_dir, output_file)
                
                # ensure .json extension
                if not output_file.lower().endswith('.json'):
                    output_file += '.json'
                    
                # Create output directory if needed
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Add metadata to the output
                output_data = {
                    "metadata": {
                        "source": args.source,
                        "model": args.model,
                        "frames_processed": frame_count, # Renamed for clarity
                        "total_objects_detected": total_objects, # Renamed for clarity
                        "export_time": datetime.now().isoformat(),
                        "avg_processing_fps": avg_fps # Renamed for clarity
                    },
                    "frames": object_coords # Contains frame data keyed by frame number (string)
                }
                
                # Export to JSON
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                print(f"Object coordinates exported to: {output_file}")
                print("Use this file as input for your path planning algorithm.")
            except Exception as e:
                print(f"Error exporting coordinates: {e}")
        
        # Model-specific advice for improving detection
        if total_objects < (frame_count * 0.5):  # Less than 0.5 objects per frame on average
            print("\n📝 Detection seems low. Try these improvements:")
            print("  - Use a larger model (--model yolo11l-obb.pt or yolo11x-obb.pt)")
            print("  - Lower the confidence threshold (--conf 0.15)")
            print("  - Increase the image size (--img-size 1280 or 1536)")
            print("  - Try test-time augmentation (--augment)")
            
        # Flicker-specific advice
        print("\n📝 To reduce detection flickering:")
        print("  - Use object tracking (--tracker bytetrack)")
        print("  - Enable class-agnostic NMS (--agnostic)")
        print("  - Lower the IoU threshold (--iou 0.5)")
        
        # Print output location with improved handling
        if not args.nosave:
            if save_dir:
                print(f"Results directory: {save_dir}")
                # Prefer the detected file path
                if final_output_path and os.path.exists(final_output_path):
                    print(f"Output saved to: {final_output_path}")
                # Otherwise fall back to the user’s requested path if it exists and is a video file
                elif args.output and os.path.exists(args.output) and Path(args.output).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    print(f"Output saved to: {args.output}")
                # Warn and list what was found if the specified path is missing or not found
                else:
                    found_videos = [f.name for f in Path(save_dir).iterdir() if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
                    if args.output:
                        print(f"Warning: Specified output {args.output} not found or not a recognized video format.")
                    if found_videos:
                        print(f"Found video(s) in {save_dir}: {found_videos}")
                    else:
                        print(f"Warning: No video output file found in {save_dir}.")
        
    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    args = parse_arguments()
    process_video(args)
