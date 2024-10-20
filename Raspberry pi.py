import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np
import cv2
from picamera2 import Picamera2
from PIL import Image

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Initialize the Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})  # Resize frames to 640x480
picam2.configure(config)
picam2.start()

# Define preprocessing transforms including normalization
preprocess = transforms.Compose([
    transforms.Resize((640, 480)),  # Resize to the same size as the camera configuration
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet standards
])

# Define a function to convert the frame for the model input
def preprocess_frame(frame):
    # Convert OpenCV image (numpy array) to PIL image
    img_pil = Image.fromarray(frame)
    
    # Apply the preprocessing pipeline (resize, tensor conversion, normalization)
    img_tensor = preprocess(img_pil)
    
    # Add a batch dimension (model expects a batch of images)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

# Define a function to draw bounding boxes and labels on the image
def draw_boxes(frame, outputs):
    labels = outputs[0]['labels'].detach().cpu().numpy()
    boxes = outputs[0]['boxes'].detach().cpu().numpy()
    scores = outputs[0]['scores'].detach().cpu().numpy()

    # Loop over the detections
    for i, box in enumerate(boxes):
        if scores[i] > 0.5:  # Only draw boxes for confident predictions
            # Convert box coordinates to integers
            box = box.astype(int)
            label = labels[i]
            score = scores[i]

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # Draw label and score
            text = f"Label: {label}, Score: {score:.2f}"
            cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Start capturing and processing frames in real-time
while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Preprocess the frame for the model
    img_tensor = preprocess_frame(frame)

    # Run Faster R-CNN inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Draw bounding boxes and labels on the frame
    frame_with_boxes = draw_boxes(frame, outputs)

    # Display the frame with detections
    cv2.imshow("Faster R-CNN Object Detection", frame_with_boxes)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
picam2.stop()
cv2.destroyAllWindows()
