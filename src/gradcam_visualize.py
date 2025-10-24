import torch
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from PIL import Image
import cv2
import logging

from models.faster_rcnn import create_cattle_detection_model  # your model factory

# ---- Load model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_cattle_detection_model(num_classes=2)
model.load_state_dict(torch.load("C:/Users/ASUS/Desktop/project 1/weights/faster_rcnn.pth", map_location=device))
#model.load_state_dict(torch.load("weights/faster_rcnn.pth", map_location=device))
model.to(device).eval()



# ---- Load image ----
image_path = "image_out/2.png"
img = Image.open(image_path).convert("RGB")
img = np.array(img).copy()    # fixes negative stride
img = Image.fromarray(img)    # back to PIL Image

img_np = np.array(img)
img_float = img_np.astype(np.float32) / 255.0

transform = transforms.Compose([
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0).to(device)

# ---- Run inference ----
outputs = model(input_tensor)

# Pick target class (first detection’s label)
if len(outputs[0]['boxes']) > 0:
    target_category = outputs[0]['labels'][0].item()
else:
    target_category = None

# ---- Grad-CAM ----
def _find_last_conv(module):
    """Return the last nn.Conv2d module inside `module` or None."""
    import torch.nn as nn
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

# Pick a conv layer automatically (robust to different backbone shapes)
backbone_module = getattr(model, "backbone", None)
if backbone_module is None:
    logging.error("Model has no attribute 'backbone' — cannot locate conv layer for GradCAM")
    raise RuntimeError("No backbone on model")

last_conv = _find_last_conv(backbone_module)
if last_conv is None:
    logging.error("Could not find a Conv2d layer in model.backbone; GradCAM requires a conv layer")
    raise RuntimeError("No Conv2d found in backbone")

target_layers = [last_conv]

# Target: focus Grad-CAM on a specific detection box
targets = [FasterRCNNBoxScoreTarget(labels=outputs[0]['labels'], bounding_boxes=outputs[0]['boxes'])]

with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # Overlay CAM on image
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

cv2.imwrite("gradcam/gradcam_output.jpg", cam_image[:, :, ::-1])  # save as BGR for OpenCV
print("Saved Grad-CAM to gradcam_output.jpg")
