import torch
from src.models.vgg16_yolov8 import VGG16YOLOv8 
from torchvision import transforms
import argparse
from pathlib import Path

def test_model(visualize=False):
    """Test model predictions with optional visualization."""
    # Initialize model with config and hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = {
        'hyperparams': {
            'batch_size': 1,
            'accumulation_steps': 1,
            'box_format': 'cxcywh',
            'coord_normalizer': 448,  # Match input size
            'min_box_size': 0.01,  # Relative to image size
            'max_box_aspect': 4.0,  # More reasonable aspect ratio
            'objectness_threshold': 0.1,
            'use_gradient_checkpointing': False,
            'use_grid_priors': True,
            'use_focal_loss': True
        }
    }
    model = VGG16YOLOv8(num_classes=1, config=model_config)
    model = model.to(device)
    model.eval()
    
    # Load a real test image
    from PIL import Image
    test_images = list(Path('dataset/cattle/test/images').glob('*.jpg'))
    if not test_images:
        print("No test images found!")
        return
        
    test_img = Image.open(test_images[0])
    
    # Print original image info
    print(f"\nOriginal image size: {test_img.size}")
    
    # Transform image
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(test_img).unsqueeze(0).to(device)
    print(f"Transformed tensor shape: {img_tensor.shape}")
    print(f"Tensor range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
    
    # Run inference
    with torch.no_grad():
        boxes, class_scores, objectness = model(img_tensor)
        
        # Debug predictions
        print(f"Boxes shape: {boxes.shape}")
        print(f"Class scores shape: {class_scores.shape}")
        print(f"Objectness shape: {objectness.shape}")
        print(f"\nFirst few boxes:\n{boxes[0, :5]}")
        print(f"\nFirst few class scores:\n{class_scores[0, :5]}")
        print(f"\nFirst few objectness scores:\n{objectness[0, :5]}")
        
    if visualize:
        # Convert predictions to image
        from PIL import Image, ImageDraw
        output_img = test_img.copy()
        draw = ImageDraw.Draw(output_img)
        
        import torchvision
        
        # Get raw predictions
        # Multiply objectness with class scores to get final confidence
        confidence = class_scores[0, :, 0] * objectness[0, :, 0]
        
        # Apply confidence threshold
        keep_mask = confidence > 0.5  # Higher threshold for more confident detections
        filtered_boxes = boxes[0][keep_mask]
        filtered_scores = confidence[keep_mask]
        
        # Convert boxes to corner format for NMS
        box_corners = torch.zeros_like(filtered_boxes)
        box_corners[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2]/2
        box_corners[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3]/2
        box_corners[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2]/2
        box_corners[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3]/2
        
        # Apply non-maximum suppression
        nms_mask = torchvision.ops.nms(
            box_corners,
            filtered_scores,
            iou_threshold=0.3  # Lower IoU for less overlap
        )
        filtered_boxes = filtered_boxes[nms_mask]
        filtered_scores = filtered_scores[nms_mask]
        
        print(f"\nAfter filtering:")
        print(f"Number of detected boxes: {len(filtered_boxes)}")
        if len(filtered_scores) > 0:
            print(f"Score range: {filtered_scores.min():.3f} - {filtered_scores.max():.3f}")
        
        # Draw surviving boxes 
        total_boxes = 0
        img_h, img_w = test_img.size[::-1]  # PIL size is (w,h)
        
        for box, score in zip(filtered_boxes, filtered_scores):
            # Get normalized coordinates (already in [0,1])
            cx, cy, w, h = box.cpu().numpy()
            
            # Convert to image coordinates
            cx = cx * img_w
            cy = cy * img_h
            w = w * img_w
            h = h * img_h
            
            # Convert to corners with clamping
            x1 = max(0, cx - w/2)
            y1 = max(0, cy - h/2)
            x2 = min(img_w, x1 + w)
            y2 = min(img_h, y1 + h)
            
            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1 or w*h < 100:  # Min box area
                continue
                
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            draw.text((x1, y1-10), f'cattle: {score:.2f}', fill='red')
            total_boxes += 1
            
            # Debug box info
            print(f"\nBox {total_boxes}:")
            print(f"Score: {score:.3f}")
            print(f"Size: {w*h:.1f} pixels ({w:.1f} x {h:.1f})")
        
        print(f"\nTotal boxes drawn: {total_boxes}")
                
        # Save output
        output_dir = Path('outputs/test_predictions')
        output_dir.mkdir(exist_ok=True, parents=True)
        output_img.save(output_dir / 'test_prediction.jpg')
        print(f"Saved visualization to {output_dir/'test_prediction.jpg'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    args = parser.parse_args()
    test_model(args.visualize)
