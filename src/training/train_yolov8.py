import os
import gc
import traceback
import psutil
import torch
import torch.cuda
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler  # Import autocast and GradScaler
from torchvision import transforms
from tqdm import tqdm
from contextlib import nullcontext
from models.yolov8 import ResNet18_YOLOv8
from processing.dataset import CattleDataset, collate_fn, worker_init_fn
from config.paths import TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS, YOLOV8_PATH
from config.hyperparameters import YOLOV8_PARAMS
from utils.memory import get_gpu_memory_usage, calculate_batch_size, log_memory_usage, clean_memory
from utils.debug import setup_debug_logging, debug_batch_dimensions
from utils.tensor_validation import validate_and_fix_tensors, validate_model_output
from utils.logging_utils import setup_logging
from utils.model_validation import validate_model_io
from utils.training_validation import validate_training_state
from utils.validation_helpers import check_model_output, validate_training_progress, validate_model_training
from utils.model_state import validate_model_state
from utils.training_safety import check_training_safety, load_checkpoint
from utils.model_parallel import setup_model_parallel
from utils.optimizer_validation import validate_optimizer
from utils.optimizer_utils import init_optimizer, validate_optimizer_state
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force GPU 0

# Initialize logger at module level
logger = setup_logging()


def validate_batch(targets):
    """Validate batch targets"""
    if not targets:
        logger.warning("Empty targets received")
        return False
    for t in targets:
        if 'boxes' not in t or 'labels' not in t:
            return False
        if len(t['boxes']) != len(t['labels']):
            return False
    return True


def validate(model, val_loader, device):
    """Improved validation with device checking"""
    model.eval()
    model_device = next(model.parameters()).device
    if model_device != device:
        logger.warning(
            f"Model device ({model_device}) doesn't match target device ({device})")
        model = model.to(device)

    total_val_loss = 0
    valid_batches = 0
    val_bar = tqdm(val_loader, desc='Validation')

    with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
        with torch.no_grad():
            for images, targets in val_bar:
                try:
                    # Validate and fix tensors
                    images, targets, is_valid = validate_and_fix_tensors(
                        images, targets)
                    if not is_valid:
                        continue

                    # Move to device
                    images = images.to(device)
                    targets = [{k: v.to(device) if torch.is_tensor(v) else v
                                for k, v in t.items()} for t in targets]

                    # Forward pass with validation
                    outputs, detections = model(images, targets)
                    if not validate_model_output(outputs):
                        continue

                    loss = model.compute_loss(outputs, targets)
                    if torch.isfinite(loss):
                        total_val_loss += loss.item()
                        valid_batches += 1
                        val_bar.set_postfix({'val_loss': loss.item()})

                except Exception as e:
                    logging.error(f"Validation error: {str(e)}")
                    continue

    if valid_batches == 0:
        logging.error("No valid batches in validation")
        return float('inf')

    return total_val_loss / valid_batches  # Avoid division by zero


def train_yolov8(**kwargs):
    try:
        # Handle device selection with new parsing
        device_arg = kwargs.get('device', 'cuda:0')
        if isinstance(device_arg, str):
            try:
                from src.utils.device_utils import parse_device
                device = parse_device(device_arg)
            except ImportError:
                logger.warning(
                    "Device utils not available, using fallback device selection")
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                else:
                    device = torch.device('cpu')
            except ValueError as e:
                logger.error(f"Invalid device specification: {e}")
                logger.info("Falling back to cuda:0 or CPU")
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                else:
                    device = torch.device('cpu')
        else:
            device = device_arg

        # Configure GPU settings with validation
        if device.type == 'cpu':
            logger.warning("Using CPU. Training will be slow!")
            gpu_props = None
            gpu_memory = 0
            actual_batch_size = 4
        else:
            try:
                # Set the specific GPU device
                if device.index is not None:
                    torch.cuda.set_device(device.index)
                else:
                    torch.cuda.set_device(0)

                torch.cuda.empty_cache()
                cudnn.benchmark = True
                cudnn.enabled = True
                gpu_props = torch.cuda.get_device_properties(device)
                gpu_memory = gpu_props.total_memory / (1024**3)
                logger.info(
                    f"Using GPU {device}: {gpu_props.name} with {gpu_memory:.1f}GB memory")
                max_batch_size = calculate_batch_size(gpu_memory)
                actual_batch_size = min(
                    max_batch_size, kwargs.get('batch_size', YOLOV8_PARAMS['batch_size']))
            except RuntimeError as e:
                logger.error(f"GPU initialization failed: {str(e)}")
                device = torch.device('cpu')
                actual_batch_size = 4

        # Calculate accumulation steps
        target_batch_size = kwargs.get(
            'batch_size', YOLOV8_PARAMS['batch_size'])
        accumulation_steps = max(
            1, target_batch_size // actual_batch_size)
        logger.info(
            f"Using batch size: {actual_batch_size} with {accumulation_steps} accumulation steps")

        print(f"Loading dataset from:")
        print(f"Train images: {TRAIN_IMAGES}")
        print(f"Train labels: {TRAIN_LABELS}")
        print(f"Val images: {VAL_IMAGES}")
        print(f"Val labels: {VAL_LABELS}")

        # Update transforms for smaller input size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((384, 384), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        try:
            # Create datasets
            train_dataset = CattleDataset(
                TRAIN_IMAGES, TRAIN_LABELS, transform=transform)
            val_dataset = CattleDataset(
                VAL_IMAGES, VAL_LABELS, transform=transform)

            print(
                f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images")

            # Configure data loaders with proper worker initialization
            num_workers = min(4, os.cpu_count() or 1)
            loader_opts = {
                'batch_size': actual_batch_size,  # Use calculated batch size
                'num_workers': num_workers,
                'pin_memory': torch.cuda.is_available(),
                'worker_init_fn': worker_init_fn,
                'persistent_workers': torch.cuda.is_available(),
                'prefetch_factor': 2 if torch.cuda.is_available() else None,
                'collate_fn': collate_fn
            }

            train_loader = DataLoader(
                train_dataset, shuffle=True, **loader_opts)
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_opts)

            # Clean memory before initialization
            torch.cuda.empty_cache()
            gc.collect()

            # Initialize GradScaler with proper device
            scaler = torch.amp.GradScaler(
                'cuda' if torch.cuda.is_available() else 'cpu')

            # Initialize model and validate
            model = ResNet18_YOLOv8(
                num_classes=1, dropout=YOLOV8_PARAMS['dropout'])
            model = model.to(device)

            # Single validation sequence
            is_valid, val_msg = validate_model_state(
                model, device, actual_batch_size)
            if not is_valid:
                raise RuntimeError(f"Model validation failed: {val_msg}")

            # Initialize optimizer with validation
            optimizer, error_msg = init_optimizer(
                model,
                YOLOV8_PARAMS['learning_rate'],
                YOLOV8_PARAMS['weight_decay']
            )
            if error_msg is not None:
                raise RuntimeError(
                    f"Optimizer initialization failed: {error_msg}")

            # Check if we can resume from checkpoint
            is_valid, checkpoint = load_checkpoint(
                YOLOV8_PATH, model, optimizer, device)
            if is_valid:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(
                        checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    best_val_loss = checkpoint['val_loss']
                    logger.info(
                        f"Resuming from epoch {start_epoch} with validation loss {best_val_loss:.4f}")
                except Exception as e:
                    logger.error(f"Failed to load checkpoint: {str(e)}")
                    start_epoch = 0
                    best_val_loss = float('inf')
            else:
                start_epoch = 0
                best_val_loss = float('inf')

            # Perform safety checks before training
            if not check_training_safety(device):
                raise RuntimeError("Training environment not safe")

            # Scheduler initialization
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5
            )

            # Print training configuration
            print(f"Training on device: {device}")
            print(f"Initial learning rate: {YOLOV8_PARAMS['learning_rate']}")

            # Add periodic memory logging
            log_memory_usage()

            # Validate model training before starting
            is_valid, message = validate_model_training(
                model, actual_batch_size, device)
            if not is_valid:
                raise RuntimeError(f"Model validation failed: {message}")

            # Validate training state before starting
            is_valid, val_msg = validate_model_state(
                model, device, actual_batch_size)
            if not is_valid:
                raise RuntimeError(f"Model validation failed: {val_msg}")

            # Continue with training state validation
            is_valid, message = validate_training_state(
                model, optimizer, device, actual_batch_size)
            if not is_valid:
                raise RuntimeError(f"Training validation failed: {message}")

            # Training loop
            patience = 10  # Early stopping patience
            patience_counter = 0

            # Update autocast configuration
            if torch.cuda.is_available():
                amp_context = torch.cuda.amp.autocast
            else:
                amp_context = nullcontext

            for epoch in range(start_epoch, kwargs.get('epochs', YOLOV8_PARAMS['num_epochs'])):
                model.train()
                total_loss = 0
                optimizer.zero_grad()
                train_bar = tqdm(
                    train_loader, desc=f'Epoch {epoch+1}/{kwargs.get("epochs", YOLOV8_PARAMS["num_epochs"])}')

                for batch_idx, (images, targets) in enumerate(train_bar):
                    try:
                        # Move to device and ensure gradients
                        images = images.to(
                            device, non_blocking=True).requires_grad_(True)
                        targets = [{k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                                    for k, v in t.items()} for t in targets]

                        # Single autocast context for entire step
                        with amp_context():
                            outputs, detections = model(images, targets)

                            if outputs is None or detections is None:
                                continue

                            loss = model.compute_loss(outputs, targets)
                            if loss is None or not torch.isfinite(loss):
                                continue

                            # Scale loss and backward pass
                            scaled_loss = loss / accumulation_steps
                            if scaler is not None:
                                scaler.scale(scaled_loss).backward()
                            else:
                                scaled_loss.backward()

                        # Update total loss
                        total_loss += scaled_loss.item() * accumulation_steps

                        # Optimizer step on accumulation boundary
                        if (batch_idx + 1) % accumulation_steps == 0:
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                                clip_grad_norm_(
                                    model.parameters(), max_norm=1.0)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                clip_grad_norm_(
                                    model.parameters(), max_norm=1.0)
                                optimizer.step()
                            optimizer.zero_grad()

                        # Debug logging after backward
                        if torch.cuda.is_available():
                            print(
                                f"[DEBUG] Memory after backward: {torch.cuda.memory_allocated()/1e9:.2f}GB")

                        # Update progress
                        train_bar.set_postfix(
                            {'loss': scaled_loss.item() * accumulation_steps})

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            optimizer.zero_grad()
                            logger.warning("GPU OOM, attempting to recover...")
                            continue
                        logger.error(
                            f"Runtime error in batch {batch_idx}: {str(e)}")
                        continue

                    except Exception as e:
                        logger.error(f"Error in batch {batch_idx}: {str(e)}")
                        continue

                avg_loss = total_loss / len(train_loader)

                # Validation and scheduler step
                val_loss = validate(model, val_loader, device)
                scheduler.step(val_loss)

                print(f"Epoch {epoch+1}/{YOLOV8_PARAMS['num_epochs']}, "
                      f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    state_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_loss,
                        'val_loss': val_loss,
                    }
                    if validate_model_state(model, state_dict, device):
                        torch.save(state_dict, YOLOV8_PATH)
                        logger.info(
                            f"Saved best model with validation loss: {val_loss:.4f}")
                    else:
                        logger.error(
                            "Failed to save model due to invalid state")

                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"Early stopping triggered after {epoch + 1} epochs")
                        break

                if epoch % 5 == 0:  # Log every 5 epochs
                    log_memory_usage()

            log_memory_usage()  # Final memory log

        except Exception as e:
            print(f"Training failed: {str(e)}")
            print(traceback.format_exc())
            if torch.cuda.is_available():
                print(
                    f"GPU Memory Status:\n{torch.cuda.memory_summary(device=None, abbreviated=False)}")
            raise

    except Exception as e:
        print(f"Critical training error: {str(e)}")
        print(traceback.format_exc())
        if torch.cuda.is_available():
            print(
                f"GPU Memory Status:\n{torch.cuda.memory_summary(device=None, abbreviated=False)}")
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
        # Call train_yolov8 with the passed arguments
        train_yolov8(**kwargs)
        return True
    except Exception as e:
        print(f"❌ YOLOv8 training failed: {str(e)}")
        return False


if __name__ == "__main__":
    train_yolov8()
