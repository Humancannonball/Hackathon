# Military Obstacle Detection System (YOLOv8-Based)

This system detects military obstacles from aerial drone images and outputs their real-world positions and risk scores for autonomous path planning. It is designed to support ground robot navigation in military settings.

## Overview

The system performs the following operations:
1. Reads aerial images from a drone
2. Recognizes ground vehicles and obstacles using YOLOv8
3. Converts YOLO bounding box outputs to relative coordinates
4. Provides obstacle coordinates and types as input to a path planning algorithm
5. Outputs JSON files with detected objects, their positions, and risk scores

## Requirements

- Python 3.8+
- Ultralytics YOLOv8
- OpenCV
- PyYAML
- NumPy

## Installation

```bash
# Install dependencies
pip install ultralytics opencv-python pyyaml numpy
```

## Usage

### Basic Usage

```bash
python military_obstacle_detector.py \
  --weights runs/train/obstacle_det/weights/best.pt \
  --source samples/*.jpg \
  --risk-config risk_map.yaml \
  --save
```

### Command Line Arguments

- `--weights`: Path to YOLOv8 weights file (required)
- `--source`: Path to source images using glob pattern (required)
- `--risk-config`: Path to risk configuration YAML file (required)
- `--save`: Optional flag to save annotated images with detection visualizations
- `--drone-altitude`: Drone altitude in meters (default: 50.0)
- `--scale-factor`: Scale factor for pixel-to-meter conversion (default: 1.0)

## Risk Configuration

The system uses a YAML file to define risk weights for different obstacle types. Example `risk_map.yaml`:

```yaml
risk_weights:
  tank: 1.0
  tree: 0.2
  trench: 0.5
  howitzer: 0.9
  apc: 0.8
  barricade: 0.6
  bunker: 0.7
  soldier: 0.4
  mine: 0.95
  fence: 0.3
```

If a detected object's class is not in the risk map, it receives a default risk value of 0.3.

## Output Format

For each processed image, the system generates a JSON file with the detected obstacles:

```json
[
  {"class": "tank", "x": 12.4, "y": 6.8, "risk": 1.0, "confidence": 0.94},
  {"class": "tree", "x": 15.1, "y": 7.3, "risk": 0.2, "confidence": 0.87}
]
```

The JSON output contains:
- `class`: Object type (e.g., tank, tree)
- `x`, `y`: Real-world ground plane coordinates in meters
- `risk`: Risk score from the risk configuration
- `confidence`: Detection confidence from YOLOv8

Output files are saved as `<image_name>.detections.json` next to the input image.

## Visual Confirmation

When the `--save` flag is used, annotated images with bounding boxes and labels are saved as `<image_name>.det.jpg`.

## Training Your Custom YOLOv8 Model

To train a custom YOLOv8 model for military obstacle detection:

1. Collect and annotate a dataset of aerial images with military obstacles
2. Organize dataset in YOLOv8 format (images and labels in appropriate directories)
3. Create a YAML file describing dataset structure and classes
4. Train the model using Ultralytics YOLOv8:

```bash
yolo task=detect mode=train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

5. Use the resulting `best.pt` weights file with the obstacle detector

## Path Planning Integration

The JSON output files are designed to be easily consumed by path planning algorithms. The risk scores can be used to adjust path costs, allowing the system to prefer paths that avoid high-risk obstacles.

## Implementation Details

### Coordinate Transformation

The script converts pixel coordinates to real-world ground coordinates using a simple scaling factor:

```python
meters_per_pixel = scale_factor * drone_altitude / image_height
```

This assumes the drone camera is pointing straight down and the ground is relatively flat.

### Risk Assessment

Each detected obstacle is assigned a risk score based on its class, as defined in the risk configuration file. This allows the path planning algorithm to calculate optimal routes that minimize risk exposure.

## Acknowledgments

- Ultralytics for the YOLOv8 framework
- OpenCV for image processing capabilities 