from torchvision import transforms
from config.paths import VAL_IMAGES, VAL_LABELS, FASTER_RCNN_PATH
from processing.dataset import CattleDataset
from models.faster_rcnn import create_cattle_detection_model
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import sys
import os

# Fix imports by adding project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_image_with_boxes(image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, thresh=0.5):
    """
    Show image with GT boxes in GREEN and predictions in RED.
    image: torch.Tensor [C,H,W]
    boxes: tensors or arrays with [xmin, ymin, xmax, ymax]
    """
    fig, ax = plt.subplots(1, figsize=(10, 8))
    img = image.permute(1, 2, 0).cpu().numpy()
    ax.imshow(img)

    # Plot ground truth (green)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    if len(gt_boxes) > 0:
        ax.text(5, 15, f"GT: {len(gt_boxes)} boxes",
                color='g', fontsize=12, weight="bold")

    # Plot predictions (red)
    for box, score in zip(pred_boxes, pred_scores):
        if score < thresh:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 5), f"{score:.2f}",
                color='r', fontsize=10, weight="bold")

    plt.axis("off")
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use transform None so dataset will convert PIL->Tensor internally
    dataset = CattleDataset(VAL_IMAGES, VAL_LABELS, transform=None)
    print(f"Validation dataset size: {len(dataset)}")

    idx = random.randint(0, len(dataset) - 1)
    image, target = dataset[idx]
    print(f"Sample index {idx}, GT labels: {target.get('labels')}")

    # Load trained model
    model = create_cattle_detection_model(num_classes=2)
    state = torch.load(FASTER_RCNN_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model([image.to(device)])[0]

    pred_boxes = output.get("boxes", torch.zeros((0, 4))).cpu()
    pred_scores = output.get(
        "scores", torch.zeros((pred_boxes.shape[0],))).cpu()
    pred_labels = output.get("labels", torch.zeros(
        (pred_boxes.shape[0],), dtype=torch.int64)).cpu()

    plot_image_with_boxes(
        image, target.get("boxes", torch.zeros((0, 4))).cpu(), target.get(
            "labels", torch.zeros((0,), dtype=torch.int64)).cpu(),
        pred_boxes, pred_scores, pred_labels,
        thresh=0.5
    )


if __name__ == "__main__":
    main()
