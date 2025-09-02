# analyze_pr_curves.py
import torch
import matplotlib.pyplot as plt
from training.train_faster_rcnn import evaluate
from processing.dataset import CattleDataset, collate_fn
from config.paths import VAL_IMAGES, VAL_LABELS, FASTER_RCNN_PATH
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from models.faster_rcnn import create_cattle_detection_model


def plot_pr_curves(results, class_names):
    if "precision" not in results or results["precision"] is None:
        return
    # [num_classes, num_thresholds, ...]
    precisions = results["precision"].cpu().numpy()
    recalls = results["recall"].cpu().numpy()

    for c, name in enumerate(class_names):
        p = precisions[c].mean(axis=0)
        r = recalls[c].mean(axis=0)
        plt.plot(r, p, label=name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-class Precision-Recall Curves")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    val_dataset = CattleDataset(VAL_IMAGES, VAL_LABELS, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=4,
                            shuffle=False, collate_fn=collate_fn)

    # Load trained model
    model = create_cattle_detection_model(num_classes=2)
    model.load_state_dict(torch.load(FASTER_RCNN_PATH, map_location=device))
    model.to(device)

    # Evaluate
    results = evaluate(model, val_loader, device)

    # Plot curves
    plot_pr_curves(results, ["background", "cattlebody"])
