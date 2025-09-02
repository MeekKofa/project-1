import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from processing.dataset import CattleDataset, collate_fn
from models.faster_rcnn import create_cattle_detection_model
from config.paths import TRAIN_IMAGES, TRAIN_LABELS
from config.hyperparameters import FASTER_RCNN_PARAMS


def main():
    img_size = FASTER_RCNN_PARAMS.get('img_size', 384)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    dataset = CattleDataset(TRAIN_IMAGES, TRAIN_LABELS, transform=transform)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    imgs, tgts = next(iter(loader))
    assert isinstance(imgs, list) and isinstance(
        tgts, list), "collate_fn must return lists"
    img = imgs[0]
    tgt = tgts[0]
    assert isinstance(
        img, torch.Tensor) and img.ndim == 3 and img.dtype == torch.float32, f"image wrong type/shape: {type(img)} {getattr(img, 'shape', None)} {getattr(img, 'dtype', None)}"
    assert 'boxes' in tgt and tgt['boxes'].shape[-1] == 4 and tgt['boxes'].dtype == torch.float32, "boxes must be float32 Nx4"
    assert tgt['labels'].dtype == torch.int64, "labels must be int64"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_cattle_detection_model(num_classes=2)
    model.to(device)
    model.train()

    imgs_cuda = [img.to(device)]
    tgt_cuda = [{k: v.to(device) for k, v in tgt.items()}]

    loss_dict = model(imgs_cuda, tgt_cuda)
    print("Loss dict:", {k: float(v) for k, v in loss_dict.items()})


if __name__ == '__main__':
    main()
