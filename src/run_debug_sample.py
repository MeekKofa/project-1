import sys
import os
import torch
import logging
from torchvision import transforms
from processing.dataset import CattleDataset
from config.paths import TRAIN_IMAGES, TRAIN_LABELS
from utils.data_validation import validate_targets, analyze_dataset_classes
from models.faster_rcnn import create_cattle_detection_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_cuda_compatibility():
    """Check CUDA availability and common issues"""
    logger.info("🔍 CUDA Compatibility Check")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    else:
        logger.warning("⚠️  CUDA not available - using CPU")


def analyze_dataset_classes_safe(dataset):
    """Safely analyze dataset classes"""
    try:
        analysis = dataset.analyze_class_distribution()
        logger.info(f"📊 Dataset Analysis: {analysis}")
        return analysis
    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}")
        return {"num_classes": 2, "error": str(e)}


def validate_sample_targets(dataset, num_samples=5):
    """Validate multiple samples and check for CUDA error causes"""
    logger.info(f"🧪 Validating {num_samples} samples...")
    issues = []

    for i in range(min(num_samples, len(dataset))):
        try:
            img, target = dataset[i]

            # Enhanced validation
            is_valid, error_msg = validate_targets([target])
            if not is_valid:
                issues.append(f"Sample {i}: {error_msg}")
                logger.error(f"❌ {error_msg}")
                continue

            # Check for specific CUDA error causes
            labels = target.get('labels', torch.tensor([]))
            boxes = target.get('boxes', torch.tensor([]))

            # Check label ranges
            if labels.numel() > 0:
                min_label, max_label = labels.min().item(), labels.max().item()
                logger.info(
                    f"✅ Sample {i}: {labels.size(0)} objects, labels: {labels.tolist()}, range: [{min_label}, {max_label}]")

                # Flag potential issues
                if min_label < 1:
                    issues.append(f"Sample {i}: Labels < 1 found: {min_label}")
                if min_label == 0:
                    issues.append(
                        f"Sample {i}: Background class (0) in object labels - should be >= 1")

            # Check box validity
            if boxes.numel() > 0:
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]

                if torch.any(widths <= 0) or torch.any(heights <= 0):
                    issues.append(f"Sample {i}: Degenerate boxes found")
                    logger.error(f"❌ Sample {i}: Invalid box dimensions")

                if torch.any(boxes < 0):
                    issues.append(f"Sample {i}: Negative coordinates found")
                    logger.error(f"❌ Sample {i}: Negative coordinates")

        except Exception as e:
            error_msg = f"Sample {i} failed: {str(e)}"
            issues.append(error_msg)
            logger.error(f"❌ {error_msg}")

    return issues


def test_model_forward_pass(num_classes, device="cuda"):
    """Test model creation and forward pass for CUDA errors"""
    logger.info(f"🤖 Testing model (num_classes={num_classes}) on {device}")

    try:
        device_obj = torch.device(
            device if torch.cuda.is_available() else "cpu")

        # Create model
        model = create_cattle_detection_model(num_classes=num_classes)
        model.to(device_obj)
        model.train()

        # Test with valid data
        # Single image, not batched
        dummy_image = torch.randn(3, 384, 384, device=device_obj)
        valid_target = {
            'boxes': torch.tensor([[10.0, 10.0, 50.0, 50.0]], device=device_obj),
            # Class 1 (valid)
            'labels': torch.tensor([1], dtype=torch.int64, device=device_obj)
        }

        logger.info("Testing with valid target...")
        with torch.no_grad():
            loss_dict = model([dummy_image], [valid_target]
                              )  # Lists of single items
            logger.info(
                f"✅ Valid target test passed: {list(loss_dict.keys())}")

        # Test with edge cases that might cause CUDA errors
        logger.info("Testing edge cases...")

        # Test with class 0 (background) - this should cause an error
        try:
            invalid_target = {
                'boxes': torch.tensor([[10.0, 10.0, 50.0, 50.0]], device=device_obj),
                # Class 0 (background)
                'labels': torch.tensor([0], dtype=torch.int64, device=device_obj)
            }
            with torch.no_grad():
                loss_dict = model([dummy_image], [invalid_target])
                logger.warning(
                    "⚠️  Background class (0) didn't cause error - this might be the issue!")
        except Exception as e:
            logger.info(f"✅ Background class correctly rejected: {str(e)}")

        # Test with out-of-range class
        try:
            invalid_target = {
                'boxes': torch.tensor([[10.0, 10.0, 50.0, 50.0]], device=device_obj),
                # >= num_classes
                'labels': torch.tensor([num_classes], dtype=torch.int64, device=device_obj)
            }
            with torch.no_grad():
                loss_dict = model([dummy_image], [invalid_target])
                logger.error(
                    f"❌ Out-of-range class {num_classes} should have caused error!")
        except Exception as e:
            logger.info(f"✅ Out-of-range class correctly rejected: {str(e)}")

        return True

    except Exception as e:
        logger.error(f"❌ Model test failed: {str(e)}")
        return False


def main(dataset_name='cattle', dataset_path=None, validate_dataset=False, sample_size=10, num_classes=None):
    """
    CUDA Error Diagnostic Tool with robust dataset configuration.

    Args:
        dataset_name (str): Name of the dataset to debug
        dataset_path (str): Direct path to dataset directory (robust mode)
        validate_dataset (bool): Whether to run comprehensive validation
        sample_size (int): Number of samples to analyze
        num_classes (int): Override for number of classes
    """
    logger.info("🚨 CUDA Error Diagnostic Tool")
    logger.info("=" * 50)

    if dataset_path:
        logger.info(f"📂 Dataset Path: {dataset_path}")
        logger.info(f"📊 Dataset Name: {dataset_name}")
    else:
        logger.info(f"📊 Dataset Name: {dataset_name} (backward compatibility)")

    # 1. Check CUDA
    check_cuda_compatibility()

    # 2. Load dataset with robust configuration
    logger.info("\n📦 Loading Dataset...")
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])

    try:
        if dataset_path:
            # Robust mode: use dataset_path directly
            train_images = os.path.join(dataset_path, 'train', 'images')
            train_labels = os.path.join(dataset_path, 'train', 'labels')

            if not os.path.exists(train_images) or not os.path.exists(train_labels):
                logger.error(
                    f"❌ Required directories not found in {dataset_path}")
                logger.error(f"   Expected: train/images and train/labels")
                sys.exit(1)

            logger.info(f"📁 Using images: {train_images}")
            logger.info(f"📁 Using labels: {train_labels}")
            ds = CattleDataset(train_images, train_labels, transform=transform)
        else:
            # Backward compatibility: use predefined paths
            ds = CattleDataset(TRAIN_IMAGES, TRAIN_LABELS, transform=transform)

        logger.info(f"Dataset size: {len(ds)}")

        # Analyze classes
        if validate_dataset:
            logger.info("🔍 Running comprehensive dataset validation...")

        class_analysis = analyze_dataset_classes_safe(ds)
        recommended_classes = num_classes or class_analysis.get(
            "num_classes", 2)
        logger.info(f"📊 Recommended num_classes: {recommended_classes}")

    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        sys.exit(1)

    # 3. Test basic sample loading (limited by sample_size)
    logger.info(
        f"\n🔍 Basic Sample Test (analyzing {min(sample_size, len(ds))} samples)...")
    test_samples = min(sample_size, len(ds))
    try:
        img, tgt = ds[0]
        logger.info(f"✅ Sample loaded successfully")
        logger.info(
            f"Image: {type(img)}, shape: {getattr(img, 'shape', None)}, dtype: {getattr(img, 'dtype', None)}")

        boxes = tgt.get("boxes", None)
        labels = tgt.get("labels", None)
        logger.info(
            f"Boxes: shape {getattr(boxes, 'shape', None)}, dtype: {getattr(boxes, 'dtype', None)}")
        logger.info(
            f"Labels: {labels.tolist() if labels is not None and labels.numel() > 0 else 'empty'}")

        # Basic assertions
        assert isinstance(img, torch.Tensor), "image must be torch.Tensor"
        assert img.ndim == 3 and img.dtype == torch.float32, "image should be [C,H,W] float32"
        assert isinstance(boxes, torch.Tensor), "boxes must be torch.Tensor"
        assert boxes.ndim == 2 and boxes.shape[1] == 4, "boxes must be shape [N,4]"
        assert isinstance(
            labels, torch.Tensor) and labels.dtype == torch.int64, "labels must be torch.int64"

        logger.info("✅ Basic format checks passed")

    except Exception as e:
        logger.error(f"❌ Basic sample test failed: {e}")
        sys.exit(1)

    # 4. Validate multiple samples
    logger.info(f"\n🧪 Sample Validation (testing {test_samples} samples)...")
    issues = validate_sample_targets(ds, test_samples)

    # 5. Test model compatibility
    logger.info(f"\n🤖 Model Compatibility Test...")
    model_success = test_model_forward_pass(recommended_classes)

    # 6. Summary
    logger.info("\n" + "=" * 50)
    logger.info("📋 DIAGNOSTIC SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Recommended num_classes: {recommended_classes}")
    logger.info(f"Dataset samples checked: {test_samples}")
    logger.info(f"Validation issues found: {len(issues)}")
    logger.info(f"Model test passed: {model_success}")

    if validate_dataset:
        logger.info("🔍 Comprehensive validation was enabled")

    if issues:
        logger.error("\n🚨 ISSUES FOUND (potential CUDA error causes):")
        for issue in issues:
            logger.error(f"  ❌ {issue}")

        logger.error("\n💡 RECOMMENDED FIXES:")
        if any("Labels < 1" in issue or "Background class" in issue for issue in issues):
            logger.error(
                "  • Check label processing in dataset.py - ensure classes are 1+ for objects")
        if any("Degenerate boxes" in issue for issue in issues):
            logger.error(
                "  • Fix bounding box coordinates in annotation files")
        if any("Negative coordinates" in issue for issue in issues):
            logger.error("  • Check coordinate system and image preprocessing")
    else:
        logger.info("✅ No critical issues found!")

    logger.info(f"\n🏁 Diagnostic complete. Exit code: {1 if issues else 0}")
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
