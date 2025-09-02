import torch
from models.yolov8 import ResNet50_YOLOv8
from torchvision import transforms

def test_model():
    # Initialize model
    model = ResNet50_YOLOv8(num_classes=1)
    model = model.to('cuda')
    model.eval()
    
    # Create dummy input
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test inference
    dummy_input = torch.randn(1, 3, 448, 448).to('cuda')
    
    with torch.no_grad():
        detections = model(dummy_input)
        
    print("Model output shape:", len(detections))
    print("Detection format:", detections[0].keys())
    
    return "Model test successful!"

if __name__ == "__main__":
    print(test_model())
