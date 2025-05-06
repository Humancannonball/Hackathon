# Drone Object Detection & Path Planning with YOLOv8 OBB

## Overview

This project enables object detection in aerial drone videos using YOLOv8 Oriented Bounding Box (OBB) models and exports detection results for downstream path planning (e.g., obstacle avoidance).

## Project Structure

```
hackathon_yolov8/
├── video.py                # Main script: runs YOLOv8 OBB detection on video, exports results to JSON
├── path_planner_interface.py  # (Optional) Converts detection JSON to occupancy/cost maps for path planning
├── README.md
```

*Delete any other files/scripts not listed above if not needed.*

## Setup

1. **Install Python 3.8+**  
   (On Fedora: `sudo dnf install python3-pip python3-venv -y`)

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install ultralytics
   ```

   *(This installs YOLOv8, PyTorch, OpenCV, and all required dependencies.)*

4. **(Optional) Check GPU availability:**
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

## Usage

### Run Object Detection on a Video

```bash
python video.py --model yolo11m-obb.pt --source your_video.mp4
```

- For best results on high-res videos, use:
  ```bash
  python video.py --model yolo11x-obb.pt --source your_video.mp4 --img-size 1536 --conf 0.15 --iou 0.5 --tracker bytetrack.yaml --agnostic --half
  ```

### Export Detection Results for Path Planning

```bash
python video.py --model yolo11m-obb.pt --source your_video.mp4 --export-coords detections.json
```

### Generate Path Planning Maps (Optional)

If you use `path_planner_interface.py`:

```bash
python path_planner_interface.py --input detections.json --visualize
```

This will create occupancy and cost maps for use with path planning algorithms.

## Tips for Best Detection

- Use a larger model (`yolo11x-obb.pt`) and higher `--img-size` for better accuracy.
- Lower `--conf` and `--iou` thresholds to detect more objects.
- Use `--tracker bytetrack.yaml` and `--agnostic` to reduce detection flicker.
- Use `--half` for faster inference on supported GPUs.

## Detectable Classes

YOLOv8 OBB models (trained on DOTA) can detect:

```
0: plane
1: ship
2: storage tank
3: baseball diamond
4: tennis court
5: basketball court
6: ground track field
7: harbor
8: bridge
9: large vehicle
10: small vehicle
11: helicopter
12: roundabout
13: soccer ball field
14: swimming pool
```

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [DOTA Dataset](https://captain-whu.github.io/DOTA/dataset.html)

---
*Keep this README and only the scripts listed in "Project Structure". Delete all other files for a clean repo.*
