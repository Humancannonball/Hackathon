#!/bin/bash
# Script to train a custom YOLOv8 model for military object detection

# Configuration
DOTA_DIR="dota"
OUTPUT_DIR="yolo_military_dataset"
MODEL_SIZE="m"  # n, s, m, l, x
EPOCHS=100
BATCH_SIZE=16
IMG_SIZE=640
MODEL_NAME="military_detector"

# Create directories
mkdir -p $DOTA_DIR/images
mkdir -p $DOTA_DIR/annotations
mkdir -p $OUTPUT_DIR

# Check dependencies
echo "Checking dependencies..."
pip install ultralytics opencv-python scikit-learn pyyaml pillow matplotlib tqdm

# Instructions for downloading DOTA dataset
echo "============================================================"
echo "STEP 1: Download the DOTA dataset"
echo "============================================================"
echo "Please download the DOTA dataset from: https://captain-whu.github.io/DOTA/dataset.html"
echo "Place image files in: $DOTA_DIR/images/"
echo "Place annotation files in: $DOTA_DIR/annotations/"
echo ""
read -p "Press Enter when you have downloaded the dataset..."

# Check if dataset exists
if [ ! "$(ls -A $DOTA_DIR/images/)" ] || [ ! "$(ls -A $DOTA_DIR/annotations/)" ]; then
    echo "Error: Dataset directories are empty. Please download the dataset first."
    exit 1
fi

# Convert dataset to YOLO format
echo "============================================================"
echo "STEP 2: Converting DOTA dataset to YOLOv8 format..."
echo "============================================================"
python dota_to_yolo.py \
  --image-dir $DOTA_DIR/images \
  --annotation-dir $DOTA_DIR/annotations \
  --output-dir $OUTPUT_DIR

# Verify conversion
if [ ! -f "$OUTPUT_DIR/dataset.yaml" ]; then
    echo "Error: Failed to convert dataset. dataset.yaml not found."
    exit 1
fi

# Train the model
echo "============================================================"
echo "STEP 3: Training the YOLOv8 model..."
echo "============================================================"
python train_military_detector.py \
  --data $OUTPUT_DIR/dataset.yaml \
  --model-size $MODEL_SIZE \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --img-size $IMG_SIZE \
  --name $MODEL_NAME

echo "============================================================"
echo "Training complete!"
echo "============================================================"
echo "The best model weights are saved in:"
echo "runs/detect/$MODEL_NAME/weights/best.pt"
echo ""
echo "To use this model with the military_obstacle_detector.py script:"
echo "python military_obstacle_detector.py \\"
echo "  --weights runs/detect/$MODEL_NAME/weights/best.pt \\"
echo "  --source test_images/*.jpg \\"
echo "  --risk-config risk_map.yaml \\"
echo "  --save"
echo "============================================================" 