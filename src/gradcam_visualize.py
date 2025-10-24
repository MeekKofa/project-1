"""Standalone Grad-CAM helper that wraps ``src.utils.gradcam_detector``."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations via project wrapper")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model", required=True, help="Model name registered in src.models.registry")
    parser.add_argument("--config", required=True, help="Dataset config YAML")
    parser.add_argument("--dataset", help="Dataset name used to resolve output paths")
    parser.add_argument("--image", help="Single image to process")
    parser.add_argument("--input-dir", help="Directory of images to process (batch mode)")
    parser.add_argument("--out", help="Output image path for single-image mode")
    parser.add_argument("--out-dir", help="Output directory for batch mode")
    parser.add_argument("--det-index", type=int, default=None, help="Detection index to target")
    parser.add_argument("--layer", help="Specific layer name for Grad-CAM")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.utils.gradcam_detector import main as gradcam_main

    forwarded_args = [
        "--checkpoint", args.checkpoint,
        "--model", args.model,
        "--config", args.config,
        "--img-size", str(args.img_size),
    ]

    if args.dataset:
        forwarded_args.extend(["--dataset", args.dataset])
    if args.image:
        forwarded_args.extend(["--image", args.image])
    if args.input_dir:
        forwarded_args.extend(["--input-dir", args.input_dir])
    if args.out:
        forwarded_args.extend(["--out", args.out])
    if args.out_dir:
        forwarded_args.extend(["--out-dir", args.out_dir])
    if args.det_index is not None:
        forwarded_args.extend(["--det-index", str(args.det_index)])
    if args.layer:
        forwarded_args.extend(["--layer", args.layer])
    if args.device:
        forwarded_args.extend(["--device", args.device])

    sys.argv = ["gradcam_visualize"] + forwarded_args
    gradcam_main()


if __name__ == "__main__":
    main()
