# Algorithms Module

This directory contains the path planning and environment mapping algorithms that will use the AI-processed drone imagery to plan optimal routes for ground robots.

## Input Format

The algorithm module will receive input from the AI module in the following JSON format:
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

## Implementation Plan

Initial implementation will focus on:

1. Creating environmental map representations from AI outputs
2. Implementing A* algorithm with terrain cost considerations
3. Developing dynamic obstacle avoidance strategies

See main README.md for further details on algorithm requirements and integration strategy.
