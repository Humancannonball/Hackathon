# Training a Custom YOLOv8 Model for Military Object Detection

This guide provides detailed instructions for training a custom YOLOv8 model specifically for detecting military objects in aerial imagery using the DOTA (Dataset for Object Detection in Aerial Images) dataset.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 16GB RAM
- 50GB+ disk space for the dataset

## Step 1: Install Required Packages

```bash
pip install ultralytics opencv-python scikit-learn pyyaml pillow matplotlib tqdm
```

## Step 2: Download and Prepare DOTA Dataset

The DOTA dataset is a large-scale aerial imagery dataset with annotated objects. You'll need to download it from the official website.

1. Visit the [DOTA dataset website](https://captain-whu.github.io/DOTA/dataset.html)
2. Register and download the dataset (both images and annotations)
3. Organize the downloaded files in the following structure:

```
dota/
├── images/
│   ├── P0001.png
│   ├── P0002.png
│   └── ...
└── annotations/
    ├── P0001.txt
    └── ...
```

## Step 3: Convert DOTA to YOLOv8 Format

The DOTA dataset uses a different annotation format than YOLOv8. Use our conversion script:

```bash
python dota_to_yolo.py \
  --image-dir dota/images \
  --annotation-dir dota/annotations \
  --output-dir yolo_military_dataset
```

This script:
1. Parses DOTA annotations
2. Filters to keep only military-relevant classes
3. Converts rotated bounding boxes to YOLO format
4. Splits data into train/val/test sets
5. Creates the necessary `dataset.yaml` file

### Customize Class Selection

To customize which classes to include, edit the `MILITARY_CLASSES` dictionary in `dota_to_yolo.py`:

```python
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
```

The DOTA dataset includes 18 classes; select the ones most relevant to your use case.

## Step 4: Train the Custom YOLOv8 Model

Use our training script to train the model:

```bash
python train_military_detector.py \
  --data yolo_military_dataset/dataset.yaml \
  --model-size m \
  --epochs 100 \
  --batch-size 16 \
  --img-size 640 \
  --name military_yolo_model
```

### Training Options

- `--model-size`: Choose from `n` (nano), `s` (small), `m` (medium), `l` (large), or `x` (xlarge)
- `--epochs`: Number of training epochs (100-300 recommended)
- `--batch-size`: Adjust based on your GPU memory
- `--img-size`: Input image size (larger = more accurate but slower)
- `--device`: Specify a CUDA device (e.g., `cuda:0`)

### Training Process

The training will:
1. Download the pre-trained YOLOv8 model as a starting point
2. Fine-tune it on your military object dataset
3. Save model checkpoints in `runs/detect/military_yolo_model/`
4. Generate performance metrics and evaluation plots

## Step 5: Monitor Training Progress

During training, you can monitor progress:

- In the terminal output
- Using TensorBoard (if installed):
  ```bash
  pip install tensorboard
  tensorboard --logdir runs/detect
  ```

## Step 6: Evaluate the Model

After training, evaluate the model on the test set:

```bash
yolo val model=runs/detect/military_yolo_model/weights/best.pt data=yolo_military_dataset/dataset.yaml
```

This will generate:
- Precision-Recall curves
- Confusion matrix
- F1-score curve
- mAP scores

## Step 7: Use the Trained Model for Inference

Once training is complete, you can use the model with our `military_obstacle_detector.py` script:

```bash
python military_obstacle_detector.py \
  --weights runs/detect/military_yolo_model/weights/best.pt \
  --source test_images/*.jpg \
  --risk-config risk_map.yaml \
  --save
```

## Optimizing Model Performance

### Data Augmentation

To improve model robustness, you can enable YOLOv8's built-in data augmentation:

```bash
python train_military_detector.py \
  --data yolo_military_dataset/dataset.yaml \
  --model-size m \
  --epochs 100 \
  --augment True
```

### Class Balancing

If your dataset is imbalanced (some classes have far more examples than others):

1. Count the instances of each class in your dataset
2. Add class weights to compensate for imbalance in the training script

### Transfer Learning vs. Training from Scratch

- For faster convergence: Use YOLOv8's pre-trained weights (default)
- For maximum customization: Train from scratch by adding `--pretrained False`

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size
   - Use a smaller model (n or s)
   - Reduce image size

2. **Slow Training**:
   - Ensure you're using GPU acceleration
   - Reduce image resolution or batch size
   - Use fewer workers

3. **Poor Detection Performance**:
   - Train for more epochs
   - Use a larger model
   - Collect more training data
   - Balance the dataset

## Advanced Options

### Hyperparameter Tuning

For better performance, try:

```bash
# Hyperparameter search
yolo task=detect mode=train model=yolov8m.pt data=yolo_military_dataset/dataset.yaml epochs=50 optimizer=Adam,SGD lr0=0.001,0.01
```

### Export for Deployment

To export the model for deployment:

```bash
# Export to ONNX format
yolo export model=runs/detect/military_yolo_model/weights/best.pt format=onnx
```

Supported formats include ONNX, TensorRT, CoreML, and more. 