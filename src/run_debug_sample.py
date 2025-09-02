import sys
import os
import torch
from torchvision import transforms
from processing.dataset import CattleDataset
from config.paths import TRAIN_IMAGES, TRAIN_LABELS


def main():
    # Minimal image transform to produce torch.Tensor [C,H,W] float32
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])

    ds = CattleDataset(TRAIN_IMAGES, TRAIN_LABELS, transform=transform)
    print("Dataset size:", len(ds))

    try:
        img, tgt = ds[0]
    except Exception as e:
        print("Failed to load sample 0:", e)
        sys.exit(1)

    print("Image type:", type(img), "shape:", getattr(
        img, "shape", None), "dtype:", getattr(img, "dtype", None))
    print("Target keys:", list(tgt.keys()))
    boxes = tgt.get("boxes", None)
    labels = tgt.get("labels", None)
    print("Boxes:", boxes)
    print("Boxes shape:", None if boxes is None else getattr(boxes, "shape", None),
          "dtype:", None if boxes is None else getattr(boxes, "dtype", None))
    print("Labels:", labels)
    print("Labels shape:", None if labels is None else getattr(labels, "shape",
          None), "dtype:", None if labels is None else getattr(labels, "dtype", None))

    # Basic assertions (will raise if format is wrong)
    assert isinstance(img, torch.Tensor), "image must be torch.Tensor"
    assert img.ndim == 3 and img.dtype == torch.float32, "image should be [C,H,W] float32"
    assert isinstance(boxes, torch.Tensor), "boxes must be torch.Tensor"
    assert boxes.ndim == 2 and boxes.shape[1] == 4, "boxes must be shape [N,4]"
    assert isinstance(
        labels, torch.Tensor) and labels.dtype == torch.int64, "labels must be torch.int64"

    print("Smoke check passed")


if __name__ == "__main__":
    main()
