"""
Visualization utilities for object detection.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Union

def visualize_detection(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    output_path: Optional[Union[str, Path]] = None,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.5
) -> None:
    """
    Visualize detection results on an image.
    
    Args:
        image: Tensor of shape [C, H, W]
        boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format, normalized if from YOLO
        labels: Tensor of shape [N] containing class indices
        output_path: Optional path to save the visualization
        class_names: Optional list of class names
        score_threshold: Minimum score threshold for visualization
    """
    # Convert image to numpy and transpose to [H, W, C]
    image = image.cpu().numpy()
    if image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    
    # Normalize image if needed
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    # Convert boxes to numpy
    boxes = boxes.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Denormalize coordinates if they are in [0, 1] range
    if boxes.max() <= 1.0:
        H, W = image.shape[:2]
        boxes = boxes.copy()  # Make a copy to avoid modifying original
        boxes[:, [0, 2]] *= W  # scale x coordinates
        boxes[:, [1, 3]] *= H  # scale y coordinates
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Plot each box
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        
        # Create rectangle patch
        rect = plt.Rectangle(
            (x1, y1), 
            x2 - x1, 
            y2 - y1,
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        plt.gca().add_patch(rect)
        
        # Add label
        class_name = class_names[label] if class_names else f'{label}'
        plt.text(
            x1, y1 - 5,
            class_name,
            bbox=dict(facecolor='red', alpha=0.5),
            fontsize=8,
            color='white'
        )
    
    plt.axis('off')
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
