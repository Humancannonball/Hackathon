# Route Planning for Ground Robots Using Drone

## Project Overview
This hackathon project develops a system where drones provide aerial surveillance to optimize ground robot navigation. The system processes drone imagery to detect obstacles and terrain features, then calculates optimal paths for ground robots to follow, with continuous updates as environmental conditions change.

## Project Directory Structure
```
Hackathon/
├── README.md                # Main project docu`mentation
├── ai/                      # AI-related code and resources
│   ├── models/              # Model definitions and weights
│   ├── data/                # Dataset storage and processing
│   └── pipeline/            # Video processing pipeline
├── algorithms/              # Path planning algorithms
│   ├── path_planning/       # Core path planning implementations
│   └── map_representation/  # Environmental map data structures
├── integration/             # Integration code connecting AI and algorithms
│   └── visualization/       # Visualization tools for demo
└── design/                  # Design documents, diagrams, and specifications
    └── system/              # System architecture diagrams
```

## Team Structure & Tasks

### AI Engineer 1
- Research and select appropriate pre-trained computer vision models
  - Recommended: YOLO family (YOLOv8) or EfficientDet for object detection
  - Alternatives: MobileNet or ResNet with SSD for lightweight implementation
- Set up model fine-tuning pipeline
- Optimize model for inference speed on available hardware
- Support integration with algorithm engineer

### AI Engineer 2
- Find and preprocess aerial imagery datasets
  - Recommended datasets:
    - [UAVid Dataset](https://uavid.nl/) - Semantic segmentation of UAV imagery
    - [Aerial Semantic Segmentation Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)
    - [DOTA](https://captain-whu.github.io/DOTA/dataset.html) - Aerial object detection
- Data annotation/augmentation if necessary
- Develop video feed processing pipeline
- Handle conversion of model outputs to algorithm inputs

### Algorithm Engineer
- Develop path planning algorithm using AI-processed data
- Implement obstacle avoidance and route optimization
- Create data structures for representing environment from vision input
- Design communication protocol between vision system and navigation system

## Technical Implementation

### Dataset Requirements
- Aerial perspective (drone view)
- Contains common obstacles (vehicles, people, structures)
- Variety of terrain types (pavement, grass, dirt, water)
- Annotations for training/validation

### Model Selection Criteria
- Fast inference time (suitable for real-time processing)
- Pre-trained on similar domains if possible
- Supports efficient fine-tuning
- Deployable on available hardware

### Data Pipeline
1. **Input**: Video frames from drone camera
2. **Processing**: AI model for object/obstacle detection and terrain classification
3. **Output Format**:
   ```json
   {
     "frame_id": 1234,
     "timestamp": "2023-11-15T14:22:36.123Z",
     "obstacles": [
       {"type": "person", "confidence": 0.92, "x1": 120, "y1": 340, "x2": 180, "y2": 480},
       {"type": "vehicle", "confidence": 0.87, "x1": 250, "y1": 100, "x2": 350, "y2": 200}
     ],
     "terrain": [
       {"type": "pavement", "confidence": 0.95, "polygon": [[0,0], [640,0], [640,120], [0,120]]},
       {"type": "grass", "confidence": 0.88, "polygon": [[100,200], [300,200], [300,400], [100,400]]}
     ]
   }
   ```

### Algorithm Requirements
- A* or RRT algorithms for path planning
- Real-time path recalculation capability
- Cost function incorporating terrain difficulty
- Safety margins around detected obstacles

## Integration Strategy
1. AI team outputs detection results in agreed JSON format
2. Algorithm team consumes these outputs to build environmental map
3. Algorithm produces waypoints for robot navigation
4. Communication protocol delivers waypoints to robot

## Quick-Start Implementation Steps

1. **Dataset Preparation**
   - Download selected dataset
   - Convert to format compatible with chosen model
   - Split into training/validation sets

2. **Model Setup & Training**
   - Install model framework (PyTorch/TensorFlow)
   - Load pre-trained weights
   - Adjust final layers for specific detection tasks
   - Fine-tune on prepared dataset

3. **Algorithm Development**
   - Create environmental representation from detection data
   - Implement path planning algorithm
   - Add obstacle avoidance logic
   - Generate waypoint instructions

4. **Integration Testing**
   - Test with recorded video samples
   - Verify detection-to-planning pipeline
   - Measure end-to-end latency

## Hackathon Demo Strategy
- Use pre-recorded drone footage if live drone unavailable
- Visualize detected obstacles and planned paths on screen
- Show path recalculation when new obstacles appear
- Demonstrate terrain-aware routing (prefer easier terrain)
