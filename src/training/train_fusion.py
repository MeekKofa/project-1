import os
import math
import argparse
import logging
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

# Local imports (user project)
from src.models.fusion_model import create_adaptive_fusion_model
from src.loaders.detection_dataset import DetectionDataset


# -------------------------- Warmup + Cosine LR Scheduler --------------------------
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine decay. Step scheduler per-batch.

    Call scheduler.step() once per optimizer.step() (i.e., per batch).
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)
        if step < self.warmup_steps and self.warmup_steps > 0:
            lr_factor = step / float(max(1, self.warmup_steps))
        else:
            progress = (step - self.warmup_steps) / float(max(1, (self.total_steps - self.warmup_steps)))
            lr_factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return [self.min_lr + (base_lr - self.min_lr) * lr_factor for base_lr in self.base_lrs]


# -------------------------- Device-aware but simplified wrapper --------------------------
class DeviceWrapper(nn.Module):
    """Simple wrapper ensuring model and inputs live on the requested device.

    This wrapper preserves the original model under `.model` so checkpoints save/load
    and other code can access the underlying module.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.model = model.to(device)
        self.device = device

    def forward(self, images, targets=None):
        # Accept images as either a batch Tensor [B,C,H,W] or list of [C,H,W] tensors
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                images = list(images.to(self.device).unbind(0))
            elif images.dim() == 3:
                images = [images.to(self.device)]
            else:
                raise ValueError(f"Unsupported image tensor dims: {images.shape}")
        elif isinstance(images, list):
            images = [img.to(self.device) if isinstance(img, torch.Tensor) else torch.tensor(img, device=self.device) for img in images]
        else:
            raise ValueError("`images` must be a Tensor or a list of Tensors")

        processed_targets = None
        if targets is not None:
            # Expect a list of dicts or list-like aligned with images
            processed_targets = []
            for t in targets:
                if t is None:
                    processed_targets.append({"boxes": torch.empty((0, 4), dtype=torch.float32, device=self.device),
                                              "labels": torch.empty((0,), dtype=torch.int64, device=self.device)})
                    continue
                boxes = t.get("boxes", torch.empty((0, 4), dtype=torch.float32))
                labels = t.get("labels", torch.empty((0,), dtype=torch.int64))
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.tensor(boxes, dtype=torch.float32)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.int64)
                boxes = boxes.to(self.device).float()
                labels = labels.to(self.device).long()
                # If boxes are in x_center,y_center,w,h normalized, user should convert in dataset
                processed_targets.append({"boxes": boxes, "labels": labels})

        return self.model(images, processed_targets)


# -------------------------- Utilities --------------------------

def setup_logging(log_dir="train_logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "cattle_training_logs.txt")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers on repeated imports
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train cattle detection model")
    parser.add_argument("--dataset_root", default="/home/john/coding/cattlebiometric/dataset/cattle")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--save_dir", default="/home/john/coding/cattlebiometric/dataset/cattle")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--inference_threshold", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


# -------------------------- Data pipeline --------------------------
def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.RandomResizedCrop(480, scale=(0.75, 1.25)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


def collate_fn(batch):
    images = []
    targets = []
    for img, tgt in batch:
        if img is None:
            continue
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        if img.dim() == 4 and img.size(0) == 1:
            img = img.squeeze(0)
        if img.dim() != 3:
            raise ValueError(f"Image must be 3D [C,H,W], got {img.shape}")
        images.append(img)

        if tgt is None or not isinstance(tgt, dict):
            targets.append({"boxes": torch.empty((0, 4), dtype=torch.float32), "labels": torch.empty((0,), dtype=torch.int64)})
        else:
            boxes = tgt.get("boxes", torch.empty((0, 4), dtype=torch.float32))
            labels = tgt.get("labels", torch.empty((0,), dtype=torch.int64))
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.tensor(boxes, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.int64)
            targets.append({"boxes": boxes, "labels": labels})

    if not images:
        return [], []
    return images, targets


def load_split_dataloaders(args, train_transform, val_test_transform, logger):
    split_paths = {
        "train": {"img": os.path.join(args.dataset_root, "train", "images"), "lbl": os.path.join(args.dataset_root, "train", "labels")},
        "val": {"img": os.path.join(args.dataset_root, "val", "images"), "lbl": os.path.join(args.dataset_root, "val", "labels")},
        "test": {"img": os.path.join(args.dataset_root, "test", "images"), "lbl": os.path.join(args.dataset_root, "test", "labels")}
    }

    for split, paths in split_paths.items():
        if not os.path.exists(paths["img"]):
            raise FileNotFoundError(f"Missing {split} images: {paths['img']}")
        if not os.path.exists(paths["lbl"]):
            raise FileNotFoundError(f"Missing {split} labels: {paths['lbl']}")

    datasets = {}
    for split, paths in split_paths.items():
        datasets[split] = DetectionDataset(images_dir=paths["img"], labels_dir=paths["lbl"], transforms=(train_transform if split=="train" else val_test_transform), image_size=480)
        if len(datasets[split]) == 0:
            raise ValueError(f"No images found in {split} images folder: {paths['img']}")

    dataloaders = {}
    for split in ["train", "val", "test"]:
        kwargs = {
            "dataset": datasets[split],
            "batch_size": args.batch_size,
            "shuffle": (split == "train"),
            "num_workers": args.num_workers,
            "collate_fn": collate_fn,
            "pin_memory": True
        }
        if args.num_workers > 0:
            kwargs["prefetch_factor"] = args.prefetch_factor
            kwargs["persistent_workers"] = True
        dataloaders[split] = DataLoader(**kwargs)

    logger.info("Split dataset loaded successfully!")
    logger.info(f"Train samples: {len(datasets['train'])} | Val samples: {len(datasets['val'])} | Test samples: {len(datasets['test'])}")
    return dataloaders["train"], dataloaders["val"], dataloaders["test"]


# -------------------------- Training & Evaluation --------------------------
def train_one_epoch(model_wrapper, model, train_loader, optimizer, scaler, scheduler, device, epoch, args, logger):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"Using learning rate: {current_lr:.6f}")

    # track classifier params for targeted clipping
    classifier_params = [p for n, p in model.named_parameters() if any(k in n for k in ("classifier", "cls_head", "classification"))]

    for batch_idx, (images, targets) in enumerate(progress_bar):
        torch.cuda.empty_cache()

        if images is None or len(images) == 0:
            logger.debug(f"Skipping empty batch {batch_idx}")
            continue

        # Forward/backward with mixed precision
        try:
            with autocast(enabled=(device.type == 'cuda')):
                loss_dict = model_wrapper(images, targets)

                # If model returns a dict of losses (Faster-RCNN style), sum them safely
                if isinstance(loss_dict, dict):
                    # sanitize NaNs
                    for k, v in list(loss_dict.items()):
                        if not isinstance(v, torch.Tensor):
                            loss_dict[k] = torch.tensor(float(v), device=device)
                        else:
                            loss_dict[k] = torch.nan_to_num(v, nan=1e-3, posinf=1e3, neginf=1e3)
                    losses = sum(v for v in loss_dict.values())
                elif isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                else:
                    raise RuntimeError("Model forward did not return losses or detections during training")

                # scale by grad accumulation
                scaled_loss = losses / float(args.grad_accum_steps)

            scaler.scale(scaled_loss).backward()

            # Step optimizer when accumulation boundary reached
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # unscale to apply gradient clipping
                scaler.unscale_(optimizer)
                if classifier_params:
                    torch.nn.utils.clip_grad_norm_(classifier_params, max_norm=0.1)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # scheduler step per batch
                scheduler.step()

            batch_loss_val = losses.item() if isinstance(losses, torch.Tensor) else float(losses)
            if not math.isnan(batch_loss_val) and not math.isinf(batch_loss_val):
                total_loss += batch_loss_val * len(images)
                valid_batches += 1

            avg_loss = total_loss / (valid_batches * args.batch_size) if valid_batches > 0 else 0.0
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({"Batch Loss": f"{batch_loss_val:.4f}", "Avg Loss": f"{avg_loss:.4f}", "LR": optimizer.param_groups[0]['lr']})

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            try:
                logger.error(f"Images type: {type(images)}, length: {len(images) if isinstance(images, list) else 'N/A'}")
                if isinstance(images, list) and len(images) > 0:
                    logger.error(f"First image shape: {images[0].shape}")
            except Exception:
                pass
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

    avg_epoch_loss = total_loss / (valid_batches * args.batch_size) if valid_batches > 0 else 0.0
    logger.info(f"Epoch {epoch+1} | Training Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss


# detection evaluation (keeps counters cumulative)
def detection_evaluation(dets, det_boxes, iou_matrix, det_labels, tgt_labels, iou_thresh, true_positives, false_positives):
    matched = set()
    if "scores" in dets:
        scores = dets["scores"]
        indices = torch.argsort(scores, descending=True)
    else:
        indices = range(len(det_boxes))

    for det_idx in indices:
        if iou_matrix.shape[1] == 0:
            false_positives += 1
            continue
        best_iou_for_det, best_tgt_idx = iou_matrix[det_idx].max(dim=0)
        if best_iou_for_det >= iou_thresh and det_labels[det_idx] == tgt_labels[best_tgt_idx] and best_tgt_idx.item() not in matched:
            true_positives += 1
            matched.add(best_tgt_idx.item())
        else:
            false_positives += 1
    return true_positives, false_positives


def evaluate_model(model_wrapper, device, dataloader, split_name, logger, threshold=0.2):
    model_wrapper.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    all_detections = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Evaluating {split_name} set")
        for images, targets in progress_bar:
            if images is None or len(images) == 0:
                continue

            if isinstance(images, torch.Tensor):
                if images.dim() == 4:
                    images = list(images.unbind(0))
                elif images.dim() == 3:
                    images = [images]

            detections = model_wrapper(images)

            batch_targets = []
            for t in targets:
                boxes = t.get("boxes", torch.empty((0, 4), dtype=torch.float32))
                labels = t.get("labels", torch.empty((0,), dtype=torch.int64))
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.tensor(boxes, dtype=torch.float32)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.int64)
                batch_targets.append({"boxes": boxes.to(device), "labels": labels.to(device)})

            # filter by score
            filtered = []
            for det in detections:
                scores = det["scores"]
                mask = scores > threshold
                filtered.append({"boxes": det["boxes"][mask].to(device), "scores": det["scores"][mask].to(device), "labels": det["labels"][mask].to(device)})

            # update metric
            metric.update(filtered, batch_targets)
            all_detections.extend(filtered)
            all_targets.extend(batch_targets)

    metrics = metric.compute()
    mAP50 = metrics["map_50"].item() if "map_50" in metrics else 0.0
    mAP = metrics["map"].item() if "map" in metrics else 0.0

    # compute precision/recall manually at IOU=0.5
    tp = 0
    fp = 0
    total_gt = 0
    iou_thresh = 0.5
    for dets, tgts in zip(all_detections, all_targets):
        det_boxes = dets["boxes"]
        det_labels = dets["labels"]
        tgt_boxes = tgts["boxes"]
        tgt_labels = tgts["labels"]
        total_gt += len(tgt_boxes)
        if len(det_boxes) == 0:
            continue
        if len(tgt_boxes) == 0:
            fp += len(det_boxes)
            continue
        iou_matrix = box_iou(det_boxes, tgt_boxes)
        tp, fp = detection_evaluation(dets, det_boxes, iou_matrix, det_labels, tgt_labels, iou_thresh, tp, fp)

    fn = total_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0

    logger.info("=" * 90)
    logger.info(f"{split_name.upper()} Set Metrics:")
    logger.info(f"mAP50 :  {mAP50:.4f} ")
    logger.info(f"mAP50-95 : {mAP:.4f}")
    logger.info(f"Precision : {precision:.4f} ")
    logger.info(f"Recall :    {recall:.4f} ")
    logger.info("=" * 90)

    metric.reset()
    return mAP50, mAP, precision, recall


def save_best_checkpoint(model_wrapper, optimizer, scaler, epoch, mAP50, save_dir, logger):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_name = "cattle_best_model.pth"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    original = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": original.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_mAP50": mAP50
    }, checkpoint_path)
    logger.info(f" Best model saved to: {checkpoint_path}")
    return checkpoint_path


# -------------------------- Main --------------------------
if __name__ == '__main__':
    args = parse_args()
    logger = setup_logging()

    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        logger.warning('GPU not detected - training on CPU')
    logger.info(f'Using device: {device}')

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Data
    train_t, val_t = get_data_transforms()
    train_loader, val_loader, test_loader = load_split_dataloaders(args, train_t, val_t, logger)

    # Model
    logger.info('Initializing adaptive fusion model...')
    base_model = create_adaptive_fusion_model(num_classes=args.num_classes)
    base_model = base_model.to(device)
    model_wrapper = DeviceWrapper(base_model, device)

    # Optimizer & Scheduler
    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model_wrapper.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineLR(optimizer, warmup_steps=args.lr_warmup_steps, total_steps=total_steps, min_lr=args.min_lr)

    start_epoch = 0
    best_val = 0.0
    stagnant = 0
    best_ckpt = None

    # Resume
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f'Checkpoint not found: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        # load compatible params only
        model_state = model_wrapper.model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
        model_state.update(filtered)
        model_wrapper.model.load_state_dict(model_state)
        start_epoch = ckpt.get('epoch', 0)
        best_val = ckpt.get('val_mAP50', 0.0)
        logger.info(f'Resumed checkpoint. Loaded {len(filtered)}/{len(state)} params. start_epoch={start_epoch}, best_val={best_val:.4f}')

    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs):
            train_loss = train_one_epoch(model_wrapper, model_wrapper.model, train_loader, optimizer, scaler, scheduler, device, epoch, args, logger)

            # Validate
            model_wrapper.eval()
            with torch.no_grad():
                val_map50, val_map, val_prec, val_rec = evaluate_model(model_wrapper, device, val_loader, 'val', logger, threshold=args.inference_threshold)
            model_wrapper.train()

            # Scheduler step on epoch end is optional (we already step per batch). Keep for compatibility by not stepping here.

            # Save best
            if val_map50 > best_val:
                best_val = val_map50
                best_ckpt = save_best_checkpoint(model_wrapper, optimizer, scaler, epoch, best_val, args.save_dir, logger)
                stagnant = 0
            else:
                stagnant += 1
                logger.info(f'No improvement: Current Val mAP50 ({val_map50:.4f}) < Best ({best_val:.4f})')
                logger.info(f'Stagant epochs: {stagnant}/{args.patience}')

            # Early stopping
            if stagnant >= args.patience:
                logger.info(f'Early stopping after {stagnant} stagnant epochs')
                break

        # Final test eval on best checkpoint
        if best_ckpt and os.path.exists(best_ckpt):
            logger.info(f'Loading best model from: {best_ckpt}')
            best = torch.load(best_ckpt, map_location=device)
            model_wrapper.model.load_state_dict(best['model_state_dict'])
        logger.info('Starting final test evaluation...')
        test_map50, test_map, test_prec, test_rec = evaluate_model(model_wrapper, device, test_loader, 'test', logger, threshold=args.inference_threshold)

        logger.info('Final Test Results:')
        logger.info(f'Test mAP50: {test_map50:.4f}')
        logger.info(f'Test mAP50-95: {test_map:.4f}')
        logger.info(f'Test Precision: {test_prec:.4f}')
        logger.info(f'Test Recall: {test_rec:.4f}')

    except Exception as e:
        logger.exception(f'Training failed with exception: {e}')
        raise
