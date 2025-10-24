#!/usr/bin/env python3
r"""
Generate Grad-CAM heatmaps for detection model outputs.

Usage examples (PowerShell):
python src/utils/gradcam_detector.py --checkpoint outputs/cattleface/fusion_model/checkpoints/best.pth --model fusion_model --config src/config/cattleface.yaml --image processed_data/cattleface/val/images/IMG.jpg --out outputs/gradcam.jpg

This script attempts to:
- load model via `src.models.registry.get_model`
- pick the last Conv2d layer by default (or use --layer)
- compute Grad-CAM for either the top detection score or for a specific bbox/class when provided
- save overlayed heatmap image

If your detection model has a non-standard output, pass --layer explicitly or adjust selection logic.
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple

# Ensure project root is on sys.path so `import src` works when running the file directly
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.registry import get_model


def find_last_conv(module: torch.nn.Module) -> Tuple[Optional[str], Optional[torch.nn.Module]]:
    # Prefer conv layers that are not part of heads (rpn, head, bbox, cls)
    last = None
    candidates = []
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            candidates.append((name, m))

    # try to pick a conv not in RPN/head if possible
    for name, m in reversed(candidates):
        lname = name.lower()
        if any(x in lname for x in ('rpn', 'head', 'bbox', 'cls', 'pred')):
            continue
        return (name, m)

    # fallback to last conv
    return candidates[-1] if candidates else (None, None)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple; the gradient for forward output is grad_out[0]
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        # prefer full backward hook when available to avoid missing grad inputs
        if hasattr(target_layer, 'register_full_backward_hook'):
            target_layer.register_full_backward_hook(backward_hook)
        else:
            target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor, target_score: torch.Tensor):
        self.model.zero_grad()
        out = self.model(input_tensor)

        # target_score may be a tensor scalar already or a callable selector
        if callable(target_score):
            score = target_score(out)
        else:
            score = target_score

        if not torch.is_tensor(score):
            score = torch.tensor(float(score), device=self.device)

        if score.dim() != 0:
            score = score.sum()

        score.backward(retain_graph=True)

        grads = self.gradients  # [B, C, H, W]
        acts = self.activations  # [B, C, H, W]

        if grads is None:
            # fallback: no gradient captured (post-processing detached outputs). Use
            # average of activations across channels as a simple saliency proxy.
            cam = acts.mean(dim=1, keepdim=True)
        else:
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def load_image(image_path: str, img_size: int = 640, device: torch.device = torch.device('cpu')):
    if image_path is None:
        raise ValueError('No image path provided')
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Cannot open image '{image_path}': {e}")
    img = img.convert('RGB')
    orig_size = img.size
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return img, tensor, orig_size


def select_target_from_output(output):
    """Try to select a scalar target score from common detector outputs."""
    # If output is a list (like torchvision detectors) pick first element
    if isinstance(output, (list, tuple)):
        first = output[0]
        if isinstance(first, dict):
            if 'scores' in first:
                scores = first['scores']
                if len(scores) == 0:
                    return torch.tensor(0., device=scores.device)
                return scores.max()
        # fallback: if tensor, return mean
        if torch.is_tensor(first):
            return first.mean()
        return torch.tensor(0.0)

    if isinstance(output, dict):
        if 'scores' in output:
            s = output['scores']
            return s.max()
        if 'logits' in output:
            logits = output['logits']
            probs = torch.softmax(logits, dim=-1)
            return probs.max()

    if torch.is_tensor(output):
        return output.mean()

    raise RuntimeError('Unrecognized model output format')


def overlay_heatmap_on_pil(img_pil: Image.Image, heatmap: np.ndarray, output_path: str, alpha: float = 0.4):
    heatmap_u8 = np.uint8(255 * heatmap)
    cmap = plt.get_cmap('jet')
    colored = cmap(heatmap_u8 / 255.0)[:, :, :3]
    colored = np.uint8(colored * 255)
    colored_img = Image.fromarray(colored).resize(img_pil.size)
    overlay = Image.blend(img_pil, colored_img, alpha=alpha)
    overlay.save(output_path)
    return output_path


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
    else:
        state = ckpt

    try:
        model.load_state_dict(state)
    except Exception:
        # try stripping 'module.' prefix
        new_state = {}
        for k, v in state.items():
            nk = k.replace('module.', '') if k.startswith('module.') else k
            new_state[nk] = v
        model.load_state_dict(new_state)

    return model


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM for detection models')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--model', default='fusion_model', help='Model name for registry')
    parser.add_argument('--config', default='src/config/cattleface.yaml', help='Dataset config YAML')
    parser.add_argument('--image', help='Path to input image')
    parser.add_argument('--input-dir', help='Directory with images to process (batch mode)')
    parser.add_argument('--out-dir', help='Directory to write outputs in batch mode (will mirror input names)')
    parser.add_argument('--recursive', action='store_true', help='Recursively scan input-dir for images')
    parser.add_argument('--det-index', type=int, default=None, help='If provided, target this detection index (0-based) for Grad-CAM')
    parser.add_argument('--layer', default=None, help='Name of conv layer to use (module.path)')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default='outputs/gradcam.jpg')
    parser.add_argument('--alpha', type=float, default=0.4)
    args = parser.parse_args()

    device = torch.device(args.device)

    # load config to find nc if present
    nc = 1
    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            nc = int(cfg.get('nc', cfg.get('num_classes', 1)))
    except Exception:
        pass

    # instantiate model (registry requires the full config dict)
    model = get_model(args.model, num_classes=nc, config=cfg)
    model = model.to(device)

    # load checkpoint
    model = load_checkpoint_into_model(model, args.checkpoint)

    # pick layer
    if args.layer:
        named = dict(model.named_modules())
        target_layer = named.get(args.layer, None)
        if target_layer is None:
            raise RuntimeError(f"Layer {args.layer} not found in model")
    else:
        name, target_layer = find_last_conv(model)
        if target_layer is None:
            raise RuntimeError('No Conv2d layer found in model')
        print(f'Using layer {name} for Grad-CAM')

    gradcam = GradCAM(model, target_layer)

    def process_single(image_path: str, out_path: str):
        img_pil, tensor, _ = load_image(image_path, img_size=args.img_size, device=device)
        model.eval()

        # Run forward WITHOUT torch.no_grad so tensors keep grad_fn for backward
        out = model(tensor)

        # If user passed det-index, try to select that detection's score
        if args.det_index is not None:
            score = None
            if isinstance(out, (list, tuple)):
                first = out[0]
                if isinstance(first, dict) and 'scores' in first:
                    s = first['scores']
                    if len(s) > args.det_index:
                        score = s[args.det_index]
            if score is None:
                score = select_target_from_output(out)
        else:
            try:
                score = select_target_from_output(out)
            except Exception:
                score = out.mean() if torch.is_tensor(out) else torch.tensor(0., device=device)

        # Ensure score is a tensor attached to the graph. If it's a detached tensor, try to
        # re-run forward without affecting state to get a connected tensor. In practice the
        # forward above should produce tensors with grad_fn unless model used no_grad.
        if torch.is_tensor(score) and not score.requires_grad:
            # attempt to make a connected scalar by re-running forward
            out = model(tensor)
            try:
                score = select_target_from_output(out)
            except Exception:
                score = score.detach().to(device).requires_grad_(True)

        cam = gradcam(tensor, score)
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        overlay_heatmap_on_pil(img_pil, cam, out_path, alpha=args.alpha)
        print('Saved Grad-CAM overlay to', out_path)

    # Batch mode
    if args.input_dir:
        in_dir = args.input_dir
        out_dir = args.out_dir or 'outputs/gradcam_batch'
        # collect image files
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        files = []
        if args.recursive:
            for root, _, filenames in os.walk(in_dir):
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in exts:
                        files.append(os.path.join(root, fn))
        else:
            for fn in os.listdir(in_dir):
                if os.path.splitext(fn)[1].lower() in exts:
                    files.append(os.path.join(in_dir, fn))

        if len(files) == 0:
            print('No images found in', in_dir)
            return

        for f in sorted(files):
            rel = os.path.relpath(f, in_dir)
            out_path = os.path.join(out_dir, rel)
            # ensure output dir exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                process_single(f, out_path)
            except Exception as e:
                print('Error processing', f, ':', e)
        return

    # Single image mode
    if args.image:
        process_single(args.image, args.out)
        return

    print('Either --image or --input-dir must be provided')


if __name__ == '__main__':
    main()
