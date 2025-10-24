"""
Additional transforms for improving small object detection.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from typing import Dict, Tuple, List, Optional, Union, Any
import random

# Handle different albumentations versions
try:
    # For newer versions (2.0+)
    from albumentations.augmentations.geometric.functional import scale_bbox
except ImportError:
    try:
        # For older versions
        from albumentations.augmentations.geometric.resize import scale_bbox
    except ImportError:
        # Fallback implementation if not available
        def scale_bbox(bbox: Tuple[float, float, float, float], scale_x: float, scale_y: float) -> Tuple[float, float, float, float]:
            """Scale bbox coordinates."""
            x1, y1, x2, y2 = bbox
            return (
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
            )

class AdaptiveRandomSizedCrop(DualTransform):
    """
    Adaptive random sized crop that handles small objects better.
    """
    
    def __init__(
        self,
        min_size: int,
        max_size: int,
        min_area: float = 0.0001,
        min_visibility: float = 0.1,
        p: float = 1.0
    ):
        """
        Args:
            min_size: Minimum crop size
            max_size: Maximum crop size
            min_area: Minimum relative box area to preserve
            min_visibility: Minimum box visibility after crop
            p: Probability of applying the transform
        """
        super().__init__(p=p)
        self.min_size = min_size
        self.max_size = max_size
        self.min_area = min_area
        self.min_visibility = min_visibility

    def get_box_stats(self, boxes: np.ndarray) -> Tuple[float, float]:
        """Calculate statistics of box sizes."""
        if len(boxes) == 0:
            return 0.0, 0.0
            
        # Calculate areas of all boxes
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        return float(np.median(areas)), float(np.std(areas))

    def get_params_dependent_on_targets(self, params: Dict) -> Dict:
        """Get transform parameters based on input boxes."""
        boxes = params["bboxes"]
        
        if len(boxes) == 0:
            # No boxes - use default params
            crop_width = crop_height = np.random.randint(self.min_size, self.max_size + 1)
            x_min = y_min = 0
            return {"crop_coords": (x_min, y_min, crop_width, crop_height)}

        img_height, img_width = params["image"].shape[:2]
        median_area, area_std = self.get_box_stats(np.array(boxes))
        
        # Adjust crop size based on box statistics
        if median_area < self.min_area:
            # For very small objects, use larger crops
            crop_size = int(self.max_size * 0.8)
        elif area_std > 0.1:
            # For varied box sizes, use medium crops
            crop_size = int(np.mean([self.min_size, self.max_size]))
        else:
            # For consistent box sizes, use smaller crops
            crop_size = int(self.min_size * 1.2)
            
        crop_size = min(max(crop_size, self.min_size), self.max_size)
        
        # Try to center crop around boxes
        if np.random.random() < 0.7:  # 70% chance to center on a box
            box_idx = np.random.randint(len(boxes))
            box = boxes[box_idx]
            box_center_x = int((box[0] + box[2]) * img_width / 2)
            box_center_y = int((box[1] + box[3]) * img_height / 2)
            
            x_min = min(max(box_center_x - crop_size // 2, 0), img_width - crop_size)
            y_min = min(max(box_center_y - crop_size // 2, 0), img_height - crop_size)
        else:
            # Random crop position
            x_min = np.random.randint(0, img_width - crop_size + 1)
            y_min = np.random.randint(0, img_height - crop_size + 1)
            
        return {
            "crop_coords": (x_min, y_min, crop_size, crop_size)
        }

    def apply(self, img: np.ndarray, crop_coords: Tuple[int, int, int, int], **params) -> np.ndarray:
        """Apply the transform to the image."""
        x_min, y_min, width, height = crop_coords
        return img[y_min:y_min + height, x_min:x_min + width]

    def apply_to_bbox(self, bbox: List[float], crop_coords: Tuple[int, int, int, int], **params) -> List[float]:
        """Apply the transform to bounding boxes."""
        x_min, y_min, width, height = crop_coords
        img_height, img_width = params["image"].shape[:2]
        return scale_bbox(bbox, x_min, y_min, width, height, img_height, img_width)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Get transform initialization argument names."""
        return ("min_size", "max_size", "min_area", "min_visibility")

class CopyPaste(DualTransform):
    """
    Randomly paste small objects from the dataset into the current image.
    This can help improve detection of small objects by increasing their frequency.
    """
    
    def __init__(
        self,
        object_bank: Optional[Dict] = None,
        min_area: float = 0.0001,
        max_objects: int = 2,
        p: float = 0.3
    ):
        """
        Args:
            object_bank: Dict of {class_id: List[Tuple(image_crop, box)]}
            min_area: Minimum relative box area to consider an object small
            max_objects: Maximum number of objects to paste
            p: Probability of applying the transform
        """
        super().__init__(p=p)
        self.object_bank = object_bank or {}
        self.min_area = min_area
        self.max_objects = max_objects
        
    def update_object_bank(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        """Update the object bank with new small objects."""
        if len(boxes) == 0:
            return
            
        img_height, img_width = image.shape[:2]
        
        for box, label in zip(boxes, labels):
            # Convert normalized coords to absolute
            x_min = int(box[0] * img_width)
            y_min = int(box[1] * img_height)
            x_max = int(box[2] * img_width)
            y_max = int(box[3] * img_height)
            
            # Calculate relative area
            width = x_max - x_min
            height = y_max - y_min
            area = (width * height) / (img_width * img_height)
            
            if area < self.min_area:
                # Extract object crop
                crop = image[y_min:y_max, x_min:x_max].copy()
                
                if label not in self.object_bank:
                    self.object_bank[label] = []
                self.object_bank[label].append((crop, box.copy()))
    
    def apply(self, image: np.ndarray, small_objects: List = None, **params) -> np.ndarray:
        """Apply copy-paste augmentation."""
        if not small_objects:
            return image
            
        result = image.copy()
        img_height, img_width = image.shape[:2]
        
        for obj_img, box in small_objects:
            # Convert normalized coords to absolute
            x_min = int(box[0] * img_width)
            y_min = int(box[1] * img_height)
            x_max = int(box[2] * img_width)
            y_max = int(box[3] * img_height)
            
            # Resize object if needed
            obj_height, obj_width = obj_img.shape[:2]
            if obj_height != (y_max - y_min) or obj_width != (x_max - x_min):
                obj_img = cv2.resize(obj_img, (x_max - x_min, y_max - y_min))
            
            # Create a mask for smoother blending
            mask = np.ones(obj_img.shape[:2], dtype=np.float32)
            mask = cv2.GaussianBlur(mask, (3, 3), 1)
            
            # Paste the object
            for c in range(3):
                result[y_min:y_max, x_min:x_max, c] = (
                    obj_img[..., c] * mask + 
                    result[y_min:y_max, x_min:x_max, c] * (1 - mask)
                )
                
        return result
    
    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        if force_apply or random.random() < self.p:
            small_objects = []
            
            # Randomly select objects to paste
            n_objects = random.randint(1, self.max_objects)
            available_classes = list(self.object_bank.keys())
            
            if available_classes:
                for _ in range(n_objects):
                    class_id = random.choice(available_classes)
                    if self.object_bank[class_id]:
                        obj = random.choice(self.object_bank[class_id])
                        small_objects.append(obj)
            
            kwargs["small_objects"] = small_objects
            
        return super().__call__(*args, force_apply=force_apply, **kwargs)