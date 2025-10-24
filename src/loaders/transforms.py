"""
Data transforms for detection tasks.

Provides augmentation and preprocessing transforms.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import random
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import logging
import cv2

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed. Some augmentations will not be available.")

from .transforms_advanced import AdaptiveRandomSizedCrop, CopyPaste

logger = logging.getLogger(__name__)


class ImprovedDetectionTransforms:
    """Advanced augmentation pipeline for object detection tasks."""

    @staticmethod
    def normalize_boxes(boxes: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
        """Normalize box coordinates to [0,1] range."""
        if len(boxes) == 0:
            return boxes
        norm_boxes = boxes.copy()
        norm_boxes[:, [0, 2]] /= image_width
        norm_boxes[:, [1, 3]] /= image_height
        return norm_boxes

    @staticmethod
    def denormalize_boxes(boxes: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
        """Denormalize box coordinates from [0,1] range to absolute coordinates."""
        if len(boxes) == 0:
            return boxes
        denorm_boxes = boxes.copy()
        denorm_boxes[:, [0, 2]] *= image_width
        denorm_boxes[:, [1, 3]] *= image_height
        return denorm_boxes

    @staticmethod
    def _validate_boxes(boxes: np.ndarray) -> np.ndarray:
        """Validate and filter boxes."""
        if len(boxes) == 0:
            return boxes
        
        valid_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box[:4]
            
            # Basic coordinate validation
            if not (0 <= x_min <= 1 and 0 <= y_min <= 1 and 
                   0 <= x_max <= 1 and 0 <= y_max <= 1):
                continue
                
            # Check for valid box dimensions
            width = x_max - x_min
            height = y_max - y_min
            if width <= 0 or height <= 0:
                continue
                
            # Filter out extremely small boxes
            if width * height < 0.0001:  # Minimum area threshold
                continue
                
            valid_boxes.append(box)
            
        return np.array(valid_boxes) if valid_boxes else np.zeros((0, boxes.shape[1]))

    def _get_transform_config(self, target_size: int, transform_type: str = 'train') -> dict:
        """Get albumentations transform configuration with advanced augmentations."""
        bbox_params = A.BboxParams(
            format='albumentations',
            min_area=0.0001,  # Preserve small objects
            min_visibility=0.1,  # Better handle occlusions
            label_fields=['class_labels']  # Only use one label field name
        )

        transforms_list = []

        if transform_type == 'train':
            # Advanced augmentations for training
            transforms_list.extend([
                # Basic geometric transforms with improved settings
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,  # Increased shift limit
                    scale_limit=0.25,  # Increased scale limit
                    rotate_limit=20,   # Increased rotation limit
                    border_mode=cv2.BORDER_CONSTANT,
                    value=[114, 114, 114],
                    p=0.8
                ),
                
                # Ensure consistent size with padding
                A.LongestMaxSize(max_size=target_size, p=1.0),
                A.PadIfNeeded(
                    min_height=target_size,
                    min_width=target_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=[114, 114, 114],
                    p=1.0
                ),

                # Enhanced color, intensity, and noise augmentations
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,  # Increased range
                        contrast_limit=0.3,    # Increased range
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,    # Increased range
                        sat_shift_limit=30,    # Increased range
                        val_shift_limit=30,    # Increased range
                        p=1.0
                    ),
                    A.CLAHE(clip_limit=4.0, p=1.0),  # Increased clip limit
                    A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),  # Increased noise range
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0)      # Increased blur range
                ], p=0.8),

                # Additional augmentations for robustness
                A.OneOf([
                    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, p=1.0),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, p=1.0),
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        angle_lower=0,
                        angle_upper=1,
                        num_flare_circles_lower=3,
                        num_flare_circles_upper=6,
                        p=1.0
                    ),
                    A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=1.0)
                ], p=0.4),

                # Advanced augmentations for small object detection
                A.OneOf([
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=32,
                        max_width=32,
                        min_height=8,
                        min_width=8,
                        fill_value=[114, 114, 114],
                        p=1.0
                    ),
                    A.GridDistortion(p=1.0),
                    A.ElasticTransform(
                        alpha=120,
                        sigma=120 * 0.05,
                        alpha_affine=120 * 0.03,
                        p=1.0
                    )
                ], p=0.3)
            ])
        
        # Final transforms for all modes
        transforms_list.extend([
            A.Resize(target_size, target_size, always_apply=True),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                always_apply=True
            )
        ])

        return {
            'transforms': A.Compose(
                transforms_list,
                bbox_params=bbox_params,
                additional_targets={'image': 'image'}
            ),
            'target_size': target_size
        }

    def __init__(self, target_size: int = 640, train: bool = True):
        self.target_size = target_size
        self.train = train
        
        # Get configs
        train_config = self._get_transform_config(target_size, 'train')
        val_config = self._get_transform_config(target_size, 'val')
        mosaic_config = self._get_transform_config(target_size // 2, 'mosaic')  # Half size for mosaic sections
        
        # Create transforms
        self.transform = train_config['transforms'] if train else val_config['transforms']
        if train:
            self.mosaic_transform = mosaic_config['transforms']
        
        self.train = train
        self.target_size = target_size

    def _apply_transform(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray, 
                       transform: A.Compose) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply albumentations transform with proper box handling."""
        h, w = image.shape[:2]
        
        # Skip transform if no boxes: ensure we still provide the label_fields
        # expected by BboxParams (albumentations requires label_fields keys to exist)
        if len(boxes) == 0:
            transformed = transform(image=image, bboxes=[], class_labels=[])
            return transformed['image'], boxes, labels
            
        # Normalize boxes to [0,1]
        norm_boxes = self.normalize_boxes(boxes, h, w)
        
        # Apply transform
        transformed = transform(
            image=image,
            bboxes=norm_boxes.tolist(),
            class_labels=labels.tolist()  # Match the label_fields name in BboxParams
        )
        
        # Convert back to absolute coordinates
        if transformed['bboxes']:
            transformed_boxes = np.array(transformed['bboxes'])
            transformed_boxes = self.denormalize_boxes(transformed_boxes, 
                                                     transformed['image'].shape[0],
                                                     transformed['image'].shape[1])
            transformed_labels = np.array(transformed['class_labels'])  # Updated to match the label_fields name
        else:
            transformed_boxes = np.zeros((0, 4), dtype=np.float32)
            transformed_labels = np.zeros(0, dtype=np.int64)
            
        return transformed['image'], transformed_boxes, transformed_labels

    def __call__(self, image: Image.Image, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Apply transforms to image and bounding boxes."""
        # Convert PIL Image to numpy array
        image_np = np.array(image)

        if target is None:
            # No target provided: call transform with empty bbox/label fields to satisfy label_fields
            transformed = self.transform(image=image_np, bboxes=[], class_labels=[])
            return torch.from_numpy(transformed['image'].transpose(2, 0, 1)), None

        # Initialize target_out
        target_out = None

        # Apply mosaic augmentation in training mode with 30% probability
        if self.train and random.random() < 0.3:
            image_np, target_out = self._apply_mosaic(image_np, target)
        else:
            # Apply regular transforms using helper method
            boxes_np = target['boxes'].numpy()
            labels_np = target['labels'].numpy()
            image_np, boxes_np, labels_np = self._apply_transform(image_np, boxes_np, labels_np, self.transform)
            
            # Preserve metadata while updating transformed values
            target_out = {k: v for k, v in target.items() if k not in ['boxes', 'labels']}
            target_out.update({
                'boxes': torch.tensor(boxes_np, dtype=torch.float32),
                'labels': torch.tensor(labels_np, dtype=torch.int64)
            })

        # Convert to tensor
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))

        return image_tensor, target_out

    def _apply_mosaic(self, image: np.ndarray, target: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
        """Apply mosaic augmentation by combining 4 copies of the image."""
        import cv2
        mosaic_img = np.zeros((self.target_size, self.target_size, 3), dtype=np.float32)
        mosaic_boxes = []
        mosaic_labels = []

        # Convert input boxes and labels to numpy
        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()

        # Split image into 4 parts and apply transforms
        for idx in range(4):
            # Get coordinates for this section
            x1a, y1a, x2a, y2a = self._get_mosaic_coords(idx)
            section_size = x2a - x1a

            # Transform section
            img_section, boxes_section, labels_section = self._apply_transform(
                image, boxes, labels, self.mosaic_transform
            )

            # Resize section to fit mosaic grid
            h, w = img_section.shape[:2]
            if h != section_size or w != section_size:
                img_section = cv2.resize(img_section, (section_size, section_size))
                # Adjust box coordinates for resize
                if len(boxes_section):
                    boxes_section[:, [0, 2]] *= (section_size / w)
                    boxes_section[:, [1, 3]] *= (section_size / h)

            # Place transformed image
            mosaic_img[y1a:y2a, x1a:x2a] = img_section

            # Adjust box coordinates to mosaic position
            if len(boxes_section):
                boxes_section[:, [0, 2]] += x1a
                boxes_section[:, [1, 3]] += y1a
                mosaic_boxes.append(boxes_section)
                mosaic_labels.append(labels_section)

        # Convert lists to arrays and create target dict
        if mosaic_boxes:
            mosaic_boxes = np.concatenate(mosaic_boxes, axis=0)
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
        else:
            mosaic_boxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_labels = np.zeros((0,), dtype=np.int64)

        # Preserve metadata while updating transformed values
        target_out = {k: v for k, v in target.items() if k not in ['boxes', 'labels']}
        target_out.update({
            'boxes': torch.tensor(mosaic_boxes, dtype=torch.float32),
            'labels': torch.tensor(mosaic_labels, dtype=torch.int64)
        })

        return mosaic_img, target_out

    def _get_mosaic_coords(self, idx: int) -> Tuple[int, int, int, int]:
        """Get coordinates for placing mosaic sections.
        
        Args:
            idx: Index of section (0-3, starting from top-left going clockwise)
            
        Returns:
            Tuple of (x1, y1, x2, y2) coordinates for section placement
        """
        half_size = self.target_size // 2
        
        if idx == 0:  # top left
            return 0, 0, half_size, half_size
        elif idx == 1:  # top right
            return half_size, 0, self.target_size, half_size
        elif idx == 2:  # bottom left
            return 0, half_size, half_size, self.target_size
        else:  # bottom right (idx == 3)
            return half_size, half_size, self.target_size, self.target_size


class Compose:
    """Compose multiple transforms together."""
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: Optional[Dict[str, Any]] = None) -> Tuple[Any, Optional[Dict[str, Any]]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL Image to tensor."""
    def __call__(self, image: Image.Image, target: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize image with mean and std."""
    def __init__(self, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, target: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize:
    """Resize image to target size."""
    def __init__(self, size: int = 640):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, image: Image.Image, target: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # Handle PIL Image or tensor
        if isinstance(image, Image.Image):
            orig_size = image.size
            image = F.resize(image, self.size)
        else:
            orig_size = image.shape[1::-1]  # width, height
            image = F.resize(image, self.size)

        if target is not None and 'boxes' in target:
            boxes = target['boxes']
            scale_x = self.size[0] / orig_size[0]
            scale_y = self.size[1] / orig_size[1]
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target['boxes'] = boxes

        return image, target


class RandomHorizontalFlip:
    """Randomly flip image horizontally."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, target: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if random.random() < self.p:
            if isinstance(image, Image.Image):
                width = image.width
                image = F.hflip(image)
            else:
                width = image.shape[-1]
                image = F.hflip(image)
                
            if target is not None and 'boxes' in target:
                boxes = target['boxes']
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes
        return image, target


class ColorJitter:
    """Randomly change brightness, contrast, saturation."""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, image: Image.Image, target: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        image = self.transform(image)
        return image, target


class RandomCrop:
    """Randomly crop image and adjust boxes."""
    def __init__(self, min_scale: float = 0.5):
        self.min_scale = min_scale

    def __call__(self, image: Image.Image, target: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if target is None or 'boxes' not in target:
            return image, target

        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]
            
        boxes = target['boxes']

        # Random crop parameters
        scale = random.uniform(self.min_scale, 1.0)
        new_width = int(width * scale)
        new_height = int(height * scale)

        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)

        # Crop image
        if isinstance(image, Image.Image):
            image = F.crop(image, top, left, new_height, new_width)
        else:
            image = image[top:top+new_height, left:left+new_width]

        # Adjust boxes
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left).clamp(0, new_width)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top).clamp(0, new_height)

        # Filter boxes with small area
        keep = ((boxes[:, 2] - boxes[:, 0]) > 1) & ((boxes[:, 3] - boxes[:, 1]) > 1)

        if keep.sum() == 0:
            # No valid boxes, return original
            return image, target

        target['boxes'] = boxes[keep]
        target['labels'] = target['labels'][keep]

        return image, target


def get_transforms(img_size: int = 640, train: bool = True) -> ImprovedDetectionTransforms:
    """
    Get transforms for detection tasks.
    
    Args:
        img_size: Target image size
        train: Whether to use training augmentations
        
    Returns:
        ImprovedDetectionTransforms object
    """
    return ImprovedDetectionTransforms(target_size=img_size, train=train)


def get_train_transforms(img_size: int = 640) -> ImprovedDetectionTransforms:
    """
    Get training transforms with augmentation.
    
    Args:
        img_size: Target image size
    
    Returns:
        ImprovedDetectionTransforms object
    """
    return get_transforms(img_size=img_size, train=True)


def get_val_transforms(img_size: int = 640) -> ImprovedDetectionTransforms:
    """
    Get validation transforms (no augmentation).
    
    Args:
        img_size: Target image size
    
    Returns:
        ImprovedDetectionTransforms object
    """
    return get_transforms(img_size=img_size, train=False)


def get_test_transforms(img_size: int = 640) -> ImprovedDetectionTransforms:
    """
    Get test transforms (no augmentation).
    
    Args:
        img_size: Target image size
    
    Returns:
        ImprovedDetectionTransforms object
    """
    return get_transforms(img_size=img_size, train=False)


def detection_collate_fn(batch):
    """
    Collate function for detection batches.
    
    Args:
        batch: List of (image, target) tuples
    
    Returns:
        Tuple[torch.Tensor, List[Dict]]: Batched images [B, C, H, W] and list of targets
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
        
    # Stack images
    images = torch.stack(images, dim=0)
    
    return images, targets
