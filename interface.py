import torch
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image

# Load YOLOv5 model (smallest YOLOv5s model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open webcam
cap = cv2.VideoCapture(0)

# Define augmentation and preprocessing transforms
preprocess_transform = transforms.Compose([
    transforms.Resize((640, 640)),                # Resize to 640x640 (default YOLOv5 size)
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Color jittering
    transforms.RandomHorizontalFlip(p=0.5),       # Horizontal flip with 50% probability
    transforms.RandomRotation(degrees=15),        # Random rotation between -15° and 15°
    transforms.ToTensor(),                        # Convert image to PyTorch tensor and normalize (0-1)
])

# Set up window for display
cv2.namedWindow('YOLOv5 Object Detection', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply preprocessing and augmentation
    img_transformed = preprocess_transform(img_pil)

    # YOLOv5 expects a batch of images, so we add a batch dimension
    img_batch = img_transformed.unsqueeze(0)

    # Run YOLOv5 inference (you can directly pass the raw frame without preprocessing)
    results = model(frame)  # Use the raw frame directly, YOLOv5 handles pre-processing internally

    # Render the results directly onto the frame
    img_with_boxes = np.squeeze(results.render())  # YOLOv5 renders the bounding boxes

    # Convert the result back to OpenCV format (BGR)
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)

    # Show the frame with detections
    cv2.imshow('YOLOv5 Object Detection', img_with_boxes)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
