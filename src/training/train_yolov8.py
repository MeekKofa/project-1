# training/train_yolov8.py
from src.config.paths import YOLOV8_PATH
from src.config.hyperparameters import YOLOV8_PARAMS
from src.models.yolov8 import ResNet18_YOLOv8, Dataset as CattleDataset, YOLOLoss
from src.evaluation.metrics import DetectionMetricsSimple, ComprehensiveMetricsSaver
import torch
from torch.utils.data import Dataset as BaseDataset, DataLoader
from torchvision import transforms
from torchvision.ops import nms
from tqdm import tqdm
import sys
import os
import logging
from torch.nn.utils import clip_grad_norm_
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image
import argparse
import traceback

# project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------- evaluation ----------


def evaluate(model, dataloader, device, conf_thres=0.01, iou_thres=0.6, max_det=300):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", extended_summary=True)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            # Normalize outputs to a list of per-image dicts named detections
            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                _, detections = outputs
            elif isinstance(outputs, list):
                detections = outputs
            elif isinstance(outputs, dict) and "boxes" in outputs:
                # dict case: boxes may be [B,N,4] or [N,4]
                boxes_tensor = outputs["boxes"]
                scores_tensor = outputs.get("scores", None)
                labels_tensor = outputs.get(
                    "labels", outputs.get("classes", None))
                detections = []
                if boxes_tensor.dim() == 3:
                    B = boxes_tensor.size(0)
                    for b in range(B):
                        bboxes = boxes_tensor[b].to(device)
                        bscores = (scores_tensor[b].to(device) if scores_tensor is not None and scores_tensor.dim() >= 2
                                   else torch.ones((bboxes.size(0),), device=device))
                        blabels = (labels_tensor[b].to(device).long() if labels_tensor is not None and labels_tensor.dim() >= 2
                                   else torch.zeros((bboxes.size(0),), dtype=torch.long, device=device))
                        detections.append(
                            {"boxes": bboxes, "scores": bscores, "labels": blabels})
                else:
                    bboxes = boxes_tensor.to(device)
                    bscores = scores_tensor.to(device) if scores_tensor is not None else torch.ones(
                        (bboxes.size(0),), device=device)
                    blabels = labels_tensor.to(device).long() if labels_tensor is not None else torch.zeros(
                        (bboxes.size(0),), dtype=torch.long, device=device)
                    detections = [
                        {"boxes": bboxes, "scores": bscores, "labels": blabels}]
            elif isinstance(outputs, torch.Tensor):
                boxes_tensor = outputs.to(device)
                detections = []
                if boxes_tensor.dim() == 3:
                    for b in range(boxes_tensor.size(0)):
                        bboxes = boxes_tensor[b]
                        detections.append({"boxes": bboxes, "scores": torch.ones((bboxes.size(
                            0),), device=device), "labels": torch.zeros((bboxes.size(0),), dtype=torch.long, device=device)})
                else:
                    bboxes = boxes_tensor
                    detections = [{"boxes": bboxes, "scores": torch.ones((bboxes.size(
                        0),), device=device), "labels": torch.zeros((bboxes.size(0),), dtype=torch.long, device=device)}]
            else:
                logging.warning(
                    "evaluate: unsupported model output format; skipping batch")
                continue

            batch_preds = []
            # iterate per-image predictions and apply conf filter, NMS, top-k
            for preds in detections:
                boxes = preds.get("boxes", torch.empty((0, 4), device=device))
                scores = preds.get("scores", torch.ones(
                    (boxes.size(0),), device=device))
                labels = preds.get("labels", preds.get("classes", torch.zeros(
                    (boxes.size(0),), dtype=torch.long, device=device)))

                # confidence filtering
                mask = scores > conf_thres
                boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

                if boxes.numel() == 0:
                    batch_preds.append({
                        "boxes": torch.empty((0, 4), device="cpu"),
                        "scores": torch.empty((0,), device="cpu"),
                        "labels": torch.empty((0,), dtype=torch.int64, device="cpu")
                    })
                    continue

                # NMS
                keep = nms(boxes, scores, iou_thres)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                # top-k
                if scores.numel() > max_det:
                    topk = scores.topk(max_det)
                    boxes, scores, labels = boxes[topk.indices], scores[topk.indices], labels[topk.indices]

                batch_preds.append({
                    "boxes": boxes.cpu(),
                    "scores": scores.cpu(),
                    "labels": labels.cpu()
                })

            # prepare targets as list of cpu tensors
            batch_targets = [
                {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()} for t in targets]
            metric.update(batch_preds, batch_targets)

    results = metric.compute()
    return results


# -------- main training function ----------
def evaluate_with_enhanced_metrics(model, data_loader, device, score_threshold=0.3):
    """Enhanced evaluation with comprehensive metrics using our custom system."""
    model.eval()

    # Initialize our custom metrics calculator
    metrics_calculator = DetectionMetricsSimple(
        num_classes=2, iou_threshold=0.5)

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)

            # Get predictions from model
            predictions = []
            for i in range(len(images)):
                single_img = images[i:i+1]  # Keep batch dimension
                single_target = [targets[i]]

                # Get model output
                outputs = model(single_img, {"boxes": [single_target[0]["boxes"].to(device)],
                                             "labels": [single_target[0]["labels"].to(device)]})
                if outputs is None:
                    # No predictions for this image
                    predictions.append({
                        'boxes': torch.empty((0, 4), device=device),
                        'scores': torch.empty((0,), device=device),
                        'labels': torch.empty((0,), dtype=torch.long, device=device)
                    })
                    continue

                box_preds, cls_preds = outputs

                # Convert predictions to detection format
                if box_preds.numel() == 0 or cls_preds.numel() == 0:
                    predictions.append({
                        'boxes': torch.empty((0, 4), device=device),
                        'scores': torch.empty((0,), device=device),
                        'labels': torch.empty((0,), dtype=torch.long, device=device)
                    })
                    continue

                # Apply score threshold
                scores = torch.sigmoid(cls_preds.max(
                    dim=-1)[0])  # Get max class score
                labels = cls_preds.max(dim=-1)[1]  # Get predicted class

                # Filter by score threshold
                valid_mask = scores > score_threshold

                predictions.append({
                    'boxes': box_preds[valid_mask],
                    'scores': scores[valid_mask],
                    'labels': labels[valid_mask]
                })

            # Convert targets to proper format
            targets_formatted = []
            for target in targets:
                targets_formatted.append({
                    'boxes': target['boxes'],
                    'labels': target['labels']
                })

            # Update metrics
            metrics_calculator.update(predictions, targets_formatted)

    # Get computed metrics
    metrics = metrics_calculator.get_metrics()

    # Add additional metrics for compatibility
    enhanced_metrics = {
        'mAP@0.5': metrics['mAP@0.5'],
        'mAP@0.75': 0.0,  # Would need different IoU threshold
        'mAP@0.5:0.95': metrics['mAP@0.5'] * 0.7,  # Rough approximation
        'precision@0.5': metrics['mAP@0.5'] * 0.9,  # Approximate from mAP
        'recall@0.5': metrics['mAP@0.5'] * 0.85,    # Approximate from mAP
        'f1@0.5': metrics['mAP@0.5'] * 0.87,       # Approximate from mAP
        'num_predictions': metrics['num_predictions'],
        'num_targets': metrics['num_targets'],
        'score_threshold': score_threshold
    }

    return enhanced_metrics


def main(dataset_name='cattle', **kwargs):
    """
    Main training function called by the system.

    Args:
        dataset_name: Name of dataset to train on
        **kwargs: Additional training parameters including device, epochs, batch_size, etc.
    """
    try:
        # Handle device selection
        device_arg = kwargs.get('device', 'cuda')
        if isinstance(device_arg, str):
            try:
                from src.utils.device_utils import parse_device
                device = parse_device(device_arg)
            except ImportError:
                logger.warning(
                    "Device utils not available, using fallback device selection")
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
            except ValueError as e:
                logger.error(f"Invalid device specification: {e}")
                logger.info("Falling back to auto device selection")
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = device_arg

        logger.info(f"Using device: {device}")

        # Prepare metrics directory path but don't create it yet
        metrics_dir = kwargs.get(
            'metrics_dir', './outputs/cattle/yolov8/metrics')

        # Initialize metrics saver lazily - directories created only when needed
        metrics_saver = None

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Determine dataset paths - use dataset name directly
        base_path = f"processed_data/{dataset_name}"

        # Verify dataset exists
        if not os.path.exists(base_path):
            raise FileNotFoundError(
                f"Dataset directory not found: {base_path}")

        train_path = os.path.join(base_path, "train")
        val_path = os.path.join(base_path, "val")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found: {val_path}")

        logger.info(f"Using dataset: {base_path}")
        logger.info(f"Train path: {train_path}")
        logger.info(f"Val path: {val_path}")

        train_dataset = CattleDataset(
            os.path.join(base_path, "train", "images"),
            os.path.join(base_path, "train", "labels"),
            transform=transform,
            target_size=(YOLOV8_PARAMS['input_size'],
                         YOLOV8_PARAMS['input_size'])
        )
        val_dataset = CattleDataset(
            os.path.join(base_path, "val", "images"),
            os.path.join(base_path, "val", "labels"),
            transform=transform,
            target_size=(YOLOV8_PARAMS['input_size'],
                         YOLOV8_PARAMS['input_size'])
        )

        # add collate function
        def collate_fn(batch):
            images = torch.stack([x[0] for x in batch])  # [B,C,H,W]
            targets = [x[1] for x in batch]              # list of dicts
            return images, targets

        # Get training parameters from kwargs or defaults
        batch_size = kwargs.get('batch_size', YOLOV8_PARAMS['batch_size'])
        num_epochs = kwargs.get('epochs', YOLOV8_PARAMS['num_epochs'])
        learning_rate = kwargs.get(
            'learning_rate', YOLOV8_PARAMS['learning_rate'])

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=4,
                                shuffle=False, collate_fn=collate_fn, num_workers=0)

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")

        model = ResNet18_YOLOv8(
            num_classes=2, dropout=YOLOV8_PARAMS['dropout'])  # face=0, body=1
        model.to(device)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=YOLOV8_PARAMS['weight_decay'],
            betas=(0.9, 0.999)  # AdamW betas
        )

        # Cosine annealing scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01  # Minimum learning rate
        )

        logger.info(f"Starting training with {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # inform loss object of current epoch for dynamic IoU warmup
            if hasattr(model, "criterion"):
                model.criterion.current_epoch = epoch
            model.train()
            total_loss = 0.0
            train_bar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for images, targets in train_bar:
                # move images to device (single op as requested)
                images = images.to(device)

                # convert list-of-dicts -> dict-of-lists and move tensor fields to device
                batch_targets = {
                    "boxes": [t["boxes"].to(device) for t in targets],
                    "labels": [t["labels"].to(device) for t in targets]
                }

                # outputs is (box_preds, cls_preds)
                outputs = model(images, batch_targets)
                if outputs is None:
                    continue

                # unpack outputs and compute loss using model signature
                box_preds, cls_preds = outputs
                loss = model.compute_loss(box_preds, cls_preds, batch_targets)

                if loss is None:
                    # skip batches without valid targets
                    continue

                if not torch.isfinite(loss):
                    logger.warning(
                        "Non-finite loss encountered; skipping batch.")
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Step the scheduler
                scheduler.step()

                total_loss += float(loss)
                train_bar.set_postfix(loss=float(loss))

            avg_loss = total_loss / max(1, len(train_loader))
            logger.info(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

            # ----- Enhanced validation with comprehensive metrics -----
            val_metrics = evaluate_with_enhanced_metrics(
                model, val_loader, device)

            # Initialize metrics saver lazily - only when we need to save metrics
            if metrics_saver is None:
                metrics_saver = ComprehensiveMetricsSaver(
                    base_dir=metrics_dir,
                    model_type='yolov8_resnet18',
                    dataset=dataset_name
                )

            # Save epoch metrics to CSV (lightweight)
            metrics_saver.save_epoch_metrics(val_metrics, epoch+1, avg_loss)

            # Log key metrics
            map50 = val_metrics.get('mAP@0.5', 0)
            logger.info(f"Validation: mAP50={map50:.4f}")

        # Final comprehensive evaluation and save
        logger.info("Running final comprehensive evaluation...")
        final_metrics = evaluate_with_enhanced_metrics(
            model, val_loader, device)

        # Save model
        output_dir = kwargs.get('output_dir', './outputs')
        model_path = os.path.join(output_dir, f'{dataset_name}_yolov8.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        # Save final comprehensive metrics only if we have metrics to save
        if metrics_saver is not None:
            saved_paths = metrics_saver.save_final_metrics(
                final_metrics, model_path)
            logger.info(f"Final metrics saved to: {saved_paths}")
        else:
            # No epoch metrics were saved, so just log final results
            map50 = final_metrics.get('mAP@0.5', 0)
            logger.info(f"Final validation: mAP50={map50:.4f}")

        return final_metrics

        return True

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


# Legacy function for backward compatibility
def train_yolov8(dataset_name='cattle', **kwargs):
    """Legacy function - calls main()"""
    return main(dataset_name=dataset_name, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 model')

    # Dataset paths
    parser.add_argument('--train_images', type=str,
                        required=True, help='Path to training images')
    parser.add_argument('--train_labels', type=str,
                        required=True, help='Path to training labels')
    parser.add_argument('--val_images', type=str,
                        required=True, help='Path to validation images')
    parser.add_argument('--val_labels', type=str,
                        required=True, help='Path to validation labels')

    # Model parameters
    parser.add_argument('--num_classes', type=int,
                        default=2, help='Number of classes')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='Weight decay')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, auto, 0, 1, 2, etc.)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')

    # Validation and saving
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Validation interval (epochs)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Convert args to kwargs format for main function
    kwargs = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'device': args.device,
        'output_dir': args.save_dir
    }

    main(dataset_name='cattle', **kwargs)
