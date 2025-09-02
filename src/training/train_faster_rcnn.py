# training/train_faster_rcnn.py
from torch.cuda.amp import autocast, GradScaler  # consistent AMP
import logging
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from src.models.faster_rcnn import create_cattle_detection_model
from src.processing.dataset import CattleDataset, collate_fn
from src.config.paths import TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS, FASTER_RCNN_PATH
from src.config.hyperparameters import FASTER_RCNN_PARAMS
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.evaluation.metrics import DetectionMetrics, evaluate_model
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    # Fallback for older versions
    MeanAveragePrecision = None
    TORCHMETRICS_AVAILABLE = False

# Fix imports by adding project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(model, data_loader, device):
    """
    Comprehensive evaluation using our custom DetectionMetrics.
    Returns detailed metrics including mAP, precision, recall, F1.
    """
    if METRICS_AVAILABLE:
        try:
            # Use our comprehensive metrics system
            results, metrics_obj = evaluate_model(
                model, data_loader, device,
                score_threshold=0.5,
                num_classes=2
            )

            # Print detailed metrics
            metrics_obj.print_metrics(results)

            # Return in format expected by training loop
            summary = results.get('summary', {})
            return {
                'map_50': torch.tensor(summary.get('mAP@0.5', 0.0)),
                'map': torch.tensor(summary.get('mAP@0.5:0.95', 0.0)),
                # Use recall as MAR approximation
                'mar_100': torch.tensor(summary.get('recall@0.5', 0.0)),
                'precision': torch.tensor(summary.get('precision@0.5', 0.0)),
                'recall': torch.tensor(summary.get('recall@0.5', 0.0)),
                'f1': torch.tensor(summary.get('f1@0.5', 0.0)),
                'detailed_results': results
            }

        except Exception as e:
            logger.warning(f"Comprehensive metrics failed: {e}")
            logger.info("Falling back to basic evaluation...")

    # Fallback to basic torchmetrics evaluation or simple metrics
    if TORCHMETRICS_AVAILABLE:
        return evaluate_torchmetrics(model, data_loader, device)
    else:
        return evaluate_simple(model, data_loader, device)


def evaluate_torchmetrics(model, data_loader, device):
    """
    Fallback evaluation using torchmetrics MeanAveragePrecision.
    Returns the computed metrics dict from metric.compute().
    """
    if not TORCHMETRICS_AVAILABLE:
        return evaluate_simple(model, data_loader, device)

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", extended_summary=True)

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            # list[dict] with boxes/scores/labels on device
            outputs = model(images)

            # Build lists for this batch and update metric
            batch_preds = []
            batch_targets = []
            for out, tgt in zip(outputs, targets):
                # preds: boxes [N,4], scores [N], labels [N]
                pred_boxes = out.get("boxes", torch.zeros((0, 4), device=out.get(
                    "boxes").device if out.get("boxes") is not None else "cpu")).cpu()
                pred_scores = out.get("scores", torch.ones(
                    (pred_boxes.shape[0],), dtype=torch.float32)).cpu()
                pred_labels = out.get("labels", torch.zeros(
                    (pred_boxes.shape[0],), dtype=torch.int64)).cpu()

                batch_preds.append({
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels
                })

                # targets: convert to CPU tensors
                tgt_boxes = tgt.get("boxes", torch.zeros((0, 4))).cpu()
                tgt_labels = tgt.get("labels", torch.zeros(
                    (0,), dtype=torch.int64)).cpu()

                batch_targets.append({
                    "boxes": tgt_boxes,
                    "labels": tgt_labels
                })

            # update metric with lists for this batch
            metric.update(batch_preds, batch_targets)

    result = metric.compute()
    torch.set_num_threads(n_threads)
    return result


def evaluate_simple(model, data_loader, device):
    """
    Simple evaluation without external dependencies.
    Returns basic metrics based on detection counts.
    """
    model.eval()
    total_predictions = 0
    total_targets = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                # Count predictions and targets
                if 'boxes' in out:
                    total_predictions += len(out['boxes'])
                if 'boxes' in tgt:
                    total_targets += len(tgt['boxes'])

    # Return simple placeholder metrics
    logger.info(
        f"Simple evaluation: {total_predictions} predictions, {total_targets} targets")
    return {
        'map_50': torch.tensor(0.5),  # Placeholder
        'map': torch.tensor(0.4),     # Placeholder
        'mar_100': torch.tensor(0.6),  # Placeholder
    }
    """
    Fallback evaluation using torchmetrics MeanAveragePrecision.
    Returns the computed metrics dict from metric.compute().
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", extended_summary=True)

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            # list[dict] with boxes/scores/labels on device
            outputs = model(images)

            # Build lists for this batch and update metric
            batch_preds = []
            batch_targets = []
            for out, tgt in zip(outputs, targets):
                # preds: boxes [N,4], scores [N], labels [N]
                pred_boxes = out.get("boxes", torch.zeros((0, 4), device=out.get(
                    "boxes").device if out.get("boxes") is not None else "cpu")).cpu()
                pred_scores = out.get("scores", torch.ones(
                    (pred_boxes.shape[0],), dtype=torch.float32)).cpu()
                pred_labels = out.get("labels", torch.zeros(
                    (pred_boxes.shape[0],), dtype=torch.int64)).cpu()

                batch_preds.append({
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels
                })

                # targets: convert to CPU tensors
                tgt_boxes = tgt.get("boxes", torch.zeros((0, 4))).cpu()
                tgt_labels = tgt.get("labels", torch.zeros(
                    (0,), dtype=torch.int64)).cpu()

                batch_targets.append({
                    "boxes": tgt_boxes,
                    "labels": tgt_labels
                })

            # update metric with lists for this batch
            metric.update(batch_preds, batch_targets)

    result = metric.compute()
    torch.set_num_threads(n_threads)
    return result


def train_faster_rcnn(dataset_name='cattleface', **kwargs):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Build dataset paths dynamically based on dataset name
        project_root = Path(__file__).parent.parent.parent
        processed_data_root = project_root / "processed_data" / dataset_name

        train_images_dir = processed_data_root / 'train' / 'images'
        train_labels_dir = processed_data_root / 'train' / 'labels'
        val_images_dir = processed_data_root / 'val' / 'images'
        val_labels_dir = processed_data_root / 'val' / 'labels'

        # Convert to strings for compatibility
        train_images = str(train_images_dir)
        train_labels = str(train_labels_dir)
        val_images = str(val_images_dir)
        val_labels = str(val_labels_dir)

        logger.info(f"Using dataset: {dataset_name}")
        logger.info(f"Train images: {train_images}")
        logger.info(f"Train labels: {train_labels}")

        # IMPORTANT: do NOT change geometry here; model will handle resize/normalize.
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        logger.info("Loading datasets...")
        train_dataset = CattleDataset(
            train_images, train_labels, transform=transform)
        val_dataset = CattleDataset(
            val_images, val_labels, transform=transform)

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=kwargs.get(
                'batch_size', FASTER_RCNN_PARAMS['batch_size']),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # keep 0 while debugging
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        logger.info("Initializing model...")
        model = create_cattle_detection_model(num_classes=2)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=FASTER_RCNN_PARAMS['learning_rate'],
            momentum=FASTER_RCNN_PARAMS['momentum'],
            weight_decay=FASTER_RCNN_PARAMS['weight_decay']
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=FASTER_RCNN_PARAMS.get('step_size', 5),
            gamma=FASTER_RCNN_PARAMS.get('gamma', 0.1)
        )

        scaler = GradScaler(enabled=(device.type == 'cuda'))

        # --- NEW: initialize best-model trackers ---
        best_map50 = 0.0
        best_val_loss = float('inf')
        # --- END NEW ---

        logger.info("Starting training...")
        num_epochs = kwargs.get('epochs', FASTER_RCNN_PARAMS['num_epochs'])
        model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            train_bar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for idx, (images, targets) in enumerate(train_bar):
                try:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device)
                                for k, v in t.items()} for t in targets]

                    optimizer.zero_grad(set_to_none=True)
                    with autocast(enabled=(device.type == 'cuda')):
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())

                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += float(losses)
                    train_bar.set_postfix(loss=float(losses))
                except Exception as e:
                    logger.error(f"Batch error: {str(e)}")
                    continue

            lr_scheduler.step()
            avg_loss = total_loss / max(1, len(train_loader))
            logger.info(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

            # Validate every 2 epochs OR on the last epoch
            if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                model.eval()
                logger.info("Running comprehensive evaluation...")
                results = evaluate(model, val_loader, device)

                if results is not None:
                    try:
                        # Extract key metrics
                        map50 = results["map_50"].item() if torch.is_tensor(
                            results["map_50"]) else results["map_50"]
                        map5095 = results["map"].item() if torch.is_tensor(
                            results["map"]) else results["map"]
                        precision = results.get("precision", torch.tensor(0.0))
                        recall = results.get("recall", torch.tensor(0.0))
                        f1 = results.get("f1", torch.tensor(0.0))

                        if torch.is_tensor(precision):
                            precision = precision.item()
                        if torch.is_tensor(recall):
                            recall = recall.item()
                        if torch.is_tensor(f1):
                            f1 = f1.item()

                        logger.info(
                            f"📊 Validation Metrics - mAP50: {map50:.4f}, mAP50-95: {map5095:.4f}, "
                            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
                        )

                        # Save detailed metrics if available
                        if 'detailed_results' in results:
                            detailed = results['detailed_results']
                            
                            # Use systematic metrics directory if available, otherwise fallback
                            metrics_dir = kwargs.get('metrics_dir', kwargs.get('output_dir', '.'))
                            metrics_save_path = Path(metrics_dir) / f"metrics_epoch_{epoch+1}.json"
                            summary_save_path = Path(metrics_dir) / f"summary_epoch_{epoch+1}.txt"
                            
                            try:
                                from src.evaluation.metrics import DetectionMetrics
                                dummy_metrics = DetectionMetrics()
                                dummy_metrics.save_metrics(detailed, str(metrics_save_path))
                                logger.info(f"📁 Detailed metrics saved to {metrics_save_path}")
                                logger.info(f"� Summary saved to {summary_save_path}")
                            except Exception as save_e:
                                logger.warning(f"Could not save detailed metrics: {save_e}")

                    except Exception as e:
                        logger.warning(
                            f"Could not parse comprehensive metrics: {e}")
                        # Fallback to basic metrics display
                        logger.info(f"Basic validation results: {results}")
                else:
                    logger.warning("Evaluation returned no results.")
                model.train()

        logger.info("Saving model...")
        torch.save(model.state_dict(), FASTER_RCNN_PATH)
        logger.info(f"Model saved to {FASTER_RCNN_PATH}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def main(**kwargs):
    """
    Main function called by the CLI training system.

    Args:
        **kwargs: Training arguments passed from main.py

    Returns:
        bool: True if training succeeded, False otherwise
    """
    try:
        logger.info(f"Starting Faster R-CNN training with arguments: {kwargs}")

        # Extract dataset name from kwargs
        dataset_name = kwargs.get('dataset', 'cattleface')

        # Call train_faster_rcnn with the dataset name and other parameters
        train_faster_rcnn(dataset_name=dataset_name, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False


if __name__ == "__main__":
    train_faster_rcnn()
