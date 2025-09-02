__all__ = ['CattleDataset', 'collate_fn', 'worker_init_fn']

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import xml.etree.ElementTree as ET
import logging
from tqdm import tqdm
import re
import cv2
import numpy as np
from multiprocessing import Lock
import torchvision.transforms as transforms  # <-- Add this import
# replace direct import with robust fallback
try:
    from utils.data_validation import validate_targets
except Exception:
    import logging
    logging.warning("utils.data_validation not available — using local validate_targets fallback")

    def validate_targets(targets):
        """
        Minimal validation used as a fallback:
        - each target is a dict with 'boxes' and 'labels'
        - boxes: torch.Tensor shape [N,4] float32 (or convertible)
        - labels: torch.Tensor shape [N] int64 (or convertible)
        Returns (bool, message)
        """
        import torch
        for t in targets:
            if not isinstance(t, dict):
                return False, "target must be dict"
            if 'boxes' not in t or 'labels' not in t:
                return False, "missing 'boxes' or 'labels'"
            boxes = t['boxes']
            labels = t['labels']
            # allow numpy arrays as well
            if isinstance(boxes, (list, tuple, np.ndarray)):
                try:
                    boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
                except Exception:
                    return False, "boxes not convertible to tensor"
            if isinstance(labels, (list, tuple, np.ndarray)):
                try:
                    labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
                except Exception:
                    return False, "labels not convertible to tensor"
            if not isinstance(boxes, torch.Tensor) or not isinstance(labels, torch.Tensor):
                return False, "boxes/labels must be tensors"
            if boxes.ndim != 2 or boxes.size(1) != 4:
                return False, "boxes must have shape [N,4]"
            if labels.ndim != 1:
                return False, "labels must be 1D tensor"
            if labels.dtype != torch.int64:
                return False, "labels must be int64"
            if boxes.size(0) != labels.size(0):
                return False, "boxes and labels must have same first dimension"
        return True, ""

# Configure logging with file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'dataset.log')),
        logging.StreamHandler()
    ]
)

class CattleDataset(Dataset):
    # Add class-level lock for thread-safe caching
    _cache_lock = Lock()
    
    def __init__(self, image_dir, label_dir, transform=None, annotation_format="yolo", max_retries=3, cache_size=64):
        try:
            logging.info(f"Initializing dataset from {image_dir}")
            if not os.path.exists(image_dir):
                raise ValueError(f"Image directory not found: {image_dir}")
            if not os.path.exists(label_dir):
                raise ValueError(f"Label directory not found: {label_dir}")
                
            self.image_dir = image_dir
            self.label_dir = label_dir
            self.transform = transform
            self.annotation_format = annotation_format
            self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            self.cache_size = min(cache_size, len(self.images))  # Limit cache size to dataset size
            self.label_mapping = {}  # Initialize mapping dictionary
            self.pin_memory = torch.cuda.is_available()  # Enable memory pinning if GPU available
    
            if len(self.images) == 0:
                raise ValueError(f"No valid images found in {image_dir}")
                
            successful_matches = 0
            failed_matches = []
            skipped_files = []
            retry_count = {}
            
            # Name patterns for image-label matching
            self.patterns = [
                (r'_n_jpg\.rf\.[^.]+$', ''),  # Facebook format
                (r'_jpg\.rf\.[^.]+$', ''),    # Standard augmented
                (r'\.rf\.[^.]+$', ''),        # Simple augmented
                (r'_jpg$', '')                # Basic suffix
            ]
            
            # Build label lookup with all possible variations
            label_ext = {'coco': '.json', 'yolo': '.txt', 'voc': '.xml'}[annotation_format]
            self.label_files = {}
            
            # Create comprehensive label mappings
            for f in os.listdir(label_dir):
                if not f.endswith(label_ext):
                    continue
                base = os.path.splitext(f)[0]
                
                # Store all possible variations of the label name
                variations = [base]
                for pattern, repl in self.patterns:
                    clean_name = re.sub(pattern, repl, base)
                    if clean_name != base:
                        variations.append(clean_name)
                
                # Map all variations to this label file
                for v in variations:
                    self.label_files[v] = f
                    
            logging.info(f"Created {len(self.label_files)} label mappings from {len(os.listdir(label_dir))} files")
            
            # Verify labels with progress bar and retry mechanism
            pbar = tqdm(self.images, desc="Loading dataset")
            remaining_images = self.images.copy()
            
            while remaining_images and max_retries > 0:
                for img_name in remaining_images[:]:
                    try:
                        base_name = self._get_label_base_name(img_name)
                        label_path = os.path.join(label_dir, base_name + label_ext)
                        
                        if not os.path.exists(label_path):
                            retry_count[img_name] = retry_count.get(img_name, 0) + 1
                            if retry_count[img_name] >= max_retries:
                                failed_matches.append((img_name, label_path))
                                remaining_images.remove(img_name)
                            continue
                            
                        self.label_mapping[img_name] = label_path
                        successful_matches += 1
                        remaining_images.remove(img_name)
                        pbar.update(1)
                        
                    except Exception as e:
                        logging.warning(f"Attempt {retry_count.get(img_name, 0) + 1} failed for {img_name}: {str(e)}")
                        retry_count[img_name] = retry_count.get(img_name, 0) + 1
                        if retry_count[img_name] >= max_retries:
                            skipped_files.append(img_name)
                            remaining_images.remove(img_name)
                
                max_retries -= 1
            
            pbar.close()
            
            if len(skipped_files) > 0:
                logging.warning(f"Skipped {len(skipped_files)} files")
            
            if len(failed_matches) > len(self.images) * 0.5:  # If more than 50% failed
                logging.error(f"Too many failed matches ({len(failed_matches)} files)")
                raise ValueError("Dataset loading failed: Too many missing labels")
                
            # Update images list to only include successfully matched files
            self.images = [img for img in self.images if img in self.label_mapping]
            
            if not self.images:
                raise ValueError("No valid image-label pairs found")
                
            logging.info(f"Successfully loaded {len(self.images)} image-label pairs")
            
        except Exception as e:
            logging.error(f"Dataset initialization failed: {str(e)}")
            raise

        # Initialize per-process caching
        self._worker_id = None
        self._local_cache = {}
        self._cache_initialized = False
        self.cache_size = cache_size

    def _initialize_worker_cache(self):
        """Initialize worker-specific cache if not already done"""
        if not self._cache_initialized:
            worker_info = torch.utils.data.get_worker_info()
            self._worker_id = worker_info.id if worker_info else 0
            self._local_cache = {}
            self._cache_initialized = True

    def _configure_cache(self):
        """Configure the image cache"""
        self._local_cache = {}  # Initialize worker-specific cache

    def _normalize_name(self, name):
        """Normalize filename by removing common variations"""
        base = os.path.splitext(name)[0]
        for pattern, repl in self.patterns:
            base = re.sub(pattern, repl, base)
        return base

    def _get_label_base_name(self, img_name):
        """Helper method to get the correct label file base name"""
        try:
            # Try various name normalizations
            names_to_try = [
                os.path.splitext(img_name)[0],  # Original
                self._normalize_name(img_name),  # Normalized
            ]
            
            for name in names_to_try:
                if name in self.label_files:
                    return os.path.splitext(self.label_files[name])[0]
            
            logging.warning(f"No label match found for {img_name}")
            return names_to_try[0]
            
        except Exception as e:
            logging.error(f"Error processing {img_name}: {str(e)}")
            raise

    def _get_empty_targets(self):
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros(0, dtype=torch.int64),
            'area': torch.zeros(0, dtype=torch.float32),
            'iscrowd': torch.zeros(0, dtype=torch.int64)
        }

    def __len__(self):
        return len(self.images)
    
    def _parse_coco_annotation(self, label_path):
        with open(label_path, 'r') as f:
            data = json.load(f)
        
        boxes = []
        for annotation in data['annotations']:
            x, y, w, h = annotation['bbox']
            boxes.append([x, y, x+w, y+h])  # Convert to [xmin, ymin, xmax, ymax]
        
        return boxes
    
    def _parse_yolo_annotation(self, label_path, img_size):
        """
        Parse YOLO annotation file and convert normalized [x_center,y_center,w,h]
        into absolute [xmin,ymin,xmax,ymax] pixel coordinates.

        Args:
            label_path: path to .txt label file
            img_size: (img_width, img_height)
        Returns:
            boxes_np: np.ndarray shape [N,4], dtype float32
            labels_np: np.ndarray shape [N], dtype int64
        """
        try:
            if not os.path.exists(label_path):
                return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

            img_width, img_height = img_size
            boxes = []
            labels = []

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        # shift class id by +1 so background=0, cattle=1
                        cls_id = int(float(parts[0])) + 1
                        x_c = float(parts[1])
                        y_c = float(parts[2])
                        w_n = float(parts[3])
                        h_n = float(parts[4])

                        # Convert normalized to absolute pixels
                        x_c *= img_width
                        y_c *= img_height
                        w = w_n * img_width
                        h = h_n * img_height

                        x1 = x_c - w / 2.0
                        y1 = y_c - h / 2.0
                        x2 = x_c + w / 2.0
                        y2 = y_c + h / 2.0

                        # Clamp to image bounds
                        x1 = max(0.0, min(float(x1), float(img_width)))
                        y1 = max(0.0, min(float(y1), float(img_height)))
                        x2 = max(0.0, min(float(x2), float(img_width)))
                        y2 = max(0.0, min(float(y2), float(img_height)))

                        # Validate box (non-zero area)
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls_id)
                    except ValueError:
                        # skip malformed lines
                        continue

            if not boxes:
                return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

            return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

        except Exception as e:
            logging.error(f"Error parsing {label_path}: {e}")
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

    def _get_empty_annotation(self):
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

    def _create_empty_tensors(self):
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.int64)

    def _parse_voc_annotation(self, label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        
        boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
        
        return boxes
    
    def _uncached_load_image(self, img_path):
        """Base image loading function without caching"""
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logging.error(f"Failed to load image {img_path}: {str(e)}")
            raise

    def _load_image(self, path):
        """Thread-safe image loading with better error handling"""
        if path not in self._local_cache:
            with self._cache_lock:
                try:
                    image = Image.open(path).convert('RGB')
                    if image.size[0] < 10 or image.size[1] < 10:
                        raise ValueError(f"Image too small: {image.size}")
                    if len(self._local_cache) >= self.cache_size:
                        self._local_cache.clear()
                    self._local_cache[path] = image
                    self.error_count = 0  # Reset error count on success
                except Exception as e:
                    self.last_error = str(e)
                    self.error_count += 1
                    if self.error_count > 10:
                        raise RuntimeError(f"Too many loading errors: {self.last_error}")
                    raise
        return self._local_cache[path]

    def __getitem__(self, idx):
        """
        Returns:
            image: torch.FloatTensor [C,H,W], dtype=float32
            target: dict with keys 'boxes' (FloatTensor [N,4]) and 'labels' (LongTensor [N])
        """
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = self._load_image(img_path)
            
            # Get dimensions before transform
            img_width, img_height = image.size
            
            # Apply transforms (may return PIL or Tensor)
            if self.transform:
                try:
                    image = self.transform(image)
                except Exception as e:
                    logging.error(f"Transform failed for {img_path}: {str(e)}")
                    raise

            # Ensure image is a torch.Tensor [C,H,W] float32
            if isinstance(image, Image.Image):
                image = transforms.ToTensor()(image).float()
            elif isinstance(image, torch.Tensor):
                image = image.float()
            else:
                # fallback: convert numpy array -> tensor
                image = torch.as_tensor(np.array(image)).float()
                if image.ndim == 3 and image.shape[2] in (1, 3):
                    image = image.permute(2, 0, 1)

            base_name = self._get_label_base_name(img_name)
            label_ext = {'coco': '.json', 'yolo': '.txt', 'voc': '.xml'}[self.annotation_format]
            label_path = os.path.join(self.label_dir, base_name + label_ext)

            # Parse annotations into numpy arrays (boxes_np [N,4], labels_np [N])
            if self.annotation_format == "coco":
                boxes_np = np.array(self._parse_coco_annotation(label_path), dtype=np.float32)
                labels_np = np.ones((len(boxes_np),), dtype=np.int64) if len(boxes_np) else np.zeros((0,), dtype=np.int64)
            elif self.annotation_format == "yolo":
                boxes_np, labels_np = self._parse_yolo_annotation(label_path, (img_width, img_height))
            elif self.annotation_format == "voc":
                boxes_np = np.array(self._parse_voc_annotation(label_path), dtype=np.float32)
                labels_np = np.ones((len(boxes_np),), dtype=np.int64) if len(boxes_np) else np.zeros((0,), dtype=np.int64)
            else:
                raise ValueError(f"Unsupported annotation format: {self.annotation_format}")

            # Ensure numpy defaults if empty
            if boxes_np is None:
                boxes_np = np.zeros((0, 4), dtype=np.float32)
            if labels_np is None:
                labels_np = np.zeros((0,), dtype=np.int64)

            # Convert to tensors with correct dtypes
            try:
                boxes = torch.from_numpy(boxes_np.astype(np.float32)) if not isinstance(boxes_np, torch.Tensor) else boxes_np.to(dtype=torch.float32)
            except Exception:
                boxes = torch.zeros((0, 4), dtype=torch.float32)

            try:
                labels = torch.from_numpy(labels_np.astype(np.int64)) if not isinstance(labels_np, torch.Tensor) else labels_np.to(dtype=torch.int64)
            except Exception:
                labels = torch.zeros((0,), dtype=torch.int64)

            # Ensure shapes: single-box edgecases
            if boxes.ndim == 1 and boxes.numel() == 4:
                boxes = boxes.unsqueeze(0)
            if labels.ndim == 0 and labels.numel() > 0:
                labels = labels.unsqueeze(0)

            # Build minimal Faster R-CNN style target
            target = {
                "boxes": boxes.to(dtype=torch.float32),
                "labels": labels.to(dtype=torch.int64)
            }

            # Validate and fallback to empty if invalid
            is_valid, error_msg = validate_targets([target])
            if not is_valid:
                logging.warning(f"Invalid target at index {idx}: {error_msg}")
                target = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64)
                }

            return image, target

        except Exception as e:
            logging.error(f"Error loading item {idx}: {str(e)}")
            # Return empty default tensors (image is tensor [3,384,384])
            return torch.zeros((3, 384, 384), dtype=torch.float32), {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor([idx], dtype=torch.int64),
                'area': torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros(0, dtype=torch.int64)
            }

    def __del__(self):
        """Clear all caches"""
        if hasattr(self, '_local_cache'):
            self._local_cache.clear()

def collate_fn(batch):
    """
    Robust, minimal collate function.

    - Expects batch to be a list of (image, target) pairs.
    - Returns (list_of_images, list_of_targets).
    - Keeps empty targets as tensors: boxes shape [0,4], labels shape [0].
    - Skips malformed items (logs a warning) so zip(*) won't fail.
    """
    good_items = []
    for i, item in enumerate(batch):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            logging.warning(f"collate_fn: skipping malformed batch item index={i}")
            continue
        good_items.append(item)

    if len(good_items) == 0:
        return [], []

    images, targets = list(zip(*good_items))
    return list(images), list(targets)

def worker_init_fn(worker_id):
    """Initialize worker process"""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._initialize_worker_cache()