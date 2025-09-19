import torch
import logging
from typing import Any, Dict, List, Union


def setup_debug_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler()
        ]
    )


def inspect_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Print detailed tensor information"""
    logging.debug(f"\n{name}:")
    logging.debug(f"Shape: {tensor.shape}")
    logging.debug(f"Type: {tensor.dtype}")
    logging.debug(f"Device: {tensor.device}")
    if torch.isfinite(tensor).all():
        logging.debug(
            f"Range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    else:
        logging.debug("Warning: Tensor contains non-finite values")


def inspect_targets(targets: List[Dict[str, Any]]) -> None:
    """Inspect target dictionary contents"""
    for i, target in enumerate(targets):
        logging.debug(f"\nTarget {i}:")
        for k, v in target.items():
            if isinstance(v, torch.Tensor):
                inspect_tensor(v, f"target[{i}][{k}]")
            else:
                logging.debug(f"{k}: {type(v)}")


def inspect_predictions(predictions: List[torch.Tensor], name: str = "predictions") -> None:
    """Inspect model predictions"""
    logging.debug(f"\n{name}:")
    for i, pred in enumerate(predictions):
        inspect_tensor(pred, f"prediction[{i}]")
        if pred.dim() != 4:
            logging.error(
                f"Invalid prediction shape at index {i}: expected 4D tensor, got {pred.dim()}D")


def validate_shapes(pred: torch.Tensor, target: Dict[str, torch.Tensor]) -> bool:
    """Validate tensor shapes match expectations"""
    try:
        B, C, H, W = pred.shape
        boxes = target.get('boxes')
        if boxes is None:
            logging.error("Missing boxes in target")
            return False

        if boxes.dim() != 2 or boxes.size(1) != 4:
            logging.error(f"Invalid boxes shape: {boxes.shape}")
            return False

        return True
    except Exception as e:
        logging.error(f"Shape validation error: {str(e)}")
        return False


def debug_loss_computation(loss: torch.Tensor, pred: torch.Tensor, target: Dict[str, torch.Tensor]) -> None:
    """Debug loss computation issues"""
    logging.debug("\nLoss computation debug:")
    logging.debug(
        f"Loss value: {loss.item() if torch.isfinite(loss) else 'Non-finite'}")
    inspect_tensor(pred, "Predictions")
    for k, v in target.items():
        if isinstance(v, torch.Tensor):
            inspect_tensor(v, f"Target[{k}]")


def debug_batch_dimensions(images, targets, batch_idx):
    """Debug tensor dimensions in batch"""
    logging.debug(f"\nBatch {batch_idx} dimensions:")
    if isinstance(images, torch.Tensor):
        inspect_tensor(images, "images")
    elif isinstance(images, (list, tuple)):
        logging.debug(f"Images is {type(images)} of length {len(images)}")
        for i, img in enumerate(images):
            inspect_tensor(img, f"image[{i}]")

    if isinstance(targets, (list, dict)):
        for i, t in enumerate(targets):
            logging.debug(f"\nTarget {i}:")
            if 'boxes' in t:
                inspect_tensor(t['boxes'], f"boxes[{i}]")
            if 'labels' in t:
                inspect_tensor(t['labels'], f"labels[{i}]")


def debug_tensor_shapes(name: str, tensor: torch.Tensor) -> None:
    """Log detailed tensor shape information"""
    try:
        shape_info = f"{name} shape: {tensor.shape}"
        if tensor.dim() > 0:
            shape_info += f", dtype: {tensor.dtype}, device: {tensor.device}"
            shape_info += f", range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]"
        logging.debug(shape_info)
    except Exception as e:
        logging.error(f"Error inspecting tensor {name}: {str(e)}")


def log_model_outputs(outputs, detections, batch_idx):
    """Debug log model outputs"""
    try:
        box_preds, cls_preds = outputs
        logging.debug(f"Batch {batch_idx}:")
        logging.debug(f"Box predictions shape: {box_preds.shape}")
        logging.debug(f"Class predictions shape: {cls_preds.shape}")
        logging.debug(f"Detection boxes shape: {detections['boxes'].shape}")
        logging.debug(f"Detection scores shape: {detections['scores'].shape}")
    except Exception as e:
        logging.error(f"Error logging outputs: {str(e)}")


def check_cuda_error_conditions(targets: List[Dict[str, Any]], num_classes: int) -> List[str]:
    """
    Check for common conditions that cause CUDA device-side assert errors

    Args:
        targets: List of target dictionaries
        num_classes: Number of classes in model (including background)

    Returns:
        List of error messages found
    """
    errors = []

    for i, target in enumerate(targets):
        if 'boxes' not in target or 'labels' not in target:
            errors.append(f"Target {i}: Missing boxes or labels")
            continue

        boxes = target['boxes']
        labels = target['labels']

        # Check label ranges - common CUDA error cause
        if labels.numel() > 0:
            min_label = labels.min().item()
            max_label = labels.max().item()

            if min_label < 1:
                errors.append(f"Target {i}: Invalid label < 1: {min_label}")
            if max_label >= num_classes:
                errors.append(
                    f"Target {i}: Label >= num_classes ({num_classes}): {max_label}")

        # Check box validity - another common cause
        if boxes.numel() > 0:
            # Check for degenerate boxes
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]

            if torch.any(widths <= 0) or torch.any(heights <= 0):
                errors.append(f"Target {i}: Degenerate boxes found")

            # Check for negative coordinates
            if torch.any(boxes < 0):
                errors.append(f"Target {i}: Negative coordinates found")

            # Check for NaN/inf values
            if not torch.isfinite(boxes).all():
                errors.append(f"Target {i}: Non-finite box coordinates")

        # Check tensor properties
        if labels.dtype != torch.int64:
            errors.append(f"Target {i}: Labels not int64: {labels.dtype}")

        if boxes.dtype != torch.float32:
            errors.append(f"Target {i}: Boxes not float32: {boxes.dtype}")

    return errors


def debug_cuda_assertion(images, targets, model, device):
    """
    Debug CUDA assertion errors by testing model with targets

    Args:
        images: Batch of images
        targets: Batch of targets
        model: Model instance
        device: Device being used
    """
    logging.info("🚨 Debugging CUDA assertion error...")

    # Basic checks
    logging.info(f"Device: {device}")
    logging.info(
        f"Images type: {type(images)}, count: {len(images) if isinstance(images, list) else 'N/A'}")
    logging.info(
        f"Targets type: {type(targets)}, count: {len(targets) if isinstance(targets, list) else 'N/A'}")

    # Check model num_classes
    try:
        if hasattr(model, 'roi_heads') and hasattr(model.roi_heads, 'box_predictor'):
            model_num_classes = model.roi_heads.box_predictor.num_classes
            logging.info(f"Model num_classes: {model_num_classes}")
        else:
            logging.warning("Could not determine model num_classes")
            model_num_classes = 2  # fallback
    except Exception as e:
        logging.error(f"Error getting model num_classes: {e}")
        model_num_classes = 2

    # Check for CUDA error conditions
    cuda_errors = check_cuda_error_conditions(targets, model_num_classes)
    if cuda_errors:
        logging.error("🚨 CUDA Error Conditions Found:")
        for error in cuda_errors:
            logging.error(f"  ❌ {error}")

        logging.info("💡 Suggested fixes:")
        logging.info(
            "  • Run with CUDA_LAUNCH_BLOCKING=1 for exact error location")
        logging.info(
            "  • Check dataset label processing (ensure classes >= 1)")
        logging.info("  • Verify model num_classes matches dataset classes")
        logging.info("  • Validate all bounding boxes are non-degenerate")
    else:
        logging.info("✅ No obvious CUDA error conditions found")

    # Test individual samples
    logging.info("Testing individual samples...")
    for i, (img, tgt) in enumerate(zip(images, targets)):
        try:
            with torch.no_grad():
                single_result = model([img.to(device)], [tgt])
                logging.info(f"✅ Sample {i} passed")
        except Exception as e:
            logging.error(f"❌ Sample {i} failed: {str(e)}")

            # Detailed inspection of failed sample
            inspect_tensor(img, f"Failed image {i}")
            inspect_targets([tgt])

            # Check specific error conditions
            sample_errors = check_cuda_error_conditions(
                [tgt], model_num_classes)
            if sample_errors:
                logging.error(f"Sample {i} specific issues:")
                for error in sample_errors:
                    logging.error(f"    {error}")


def enable_cuda_debugging():
    """Enable CUDA debugging settings"""
    import os

    # Enable synchronous CUDA execution for better error messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Enable CUDA error checking
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Clear CUDA cache
        torch.cuda.empty_cache()

        logging.info("✅ CUDA debugging enabled")
        logging.info("  - CUDA_LAUNCH_BLOCKING=1 (synchronous execution)")
        logging.info("  - cuDNN deterministic mode enabled")
        logging.info("  - CUDA cache cleared")
    else:
        logging.info("CUDA not available - debugging settings skipped")
