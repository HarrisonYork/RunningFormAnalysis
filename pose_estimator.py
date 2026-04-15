from ultralytics import YOLO

# Load a pretrained YOLOv8 pose model (the 'n' stands for nano, which is fastest. You can use 's', 'm', 'l', or 'x' for better accuracy)
model = YOLO('yolo11n-pose.pt')

# Run inference on your running video
# The 'save=True' argument automatically generates an output video with the joints visualized
results = model(source='data/private/testVid.mp4', show=True, save=True, conf=0.5)

# To access the raw joint coordinates for your form analyzer later:
for frame_result in results:
    # Keypoints object containing the x, y coordinates and confidence scores
    keypoints = frame_result.keypoints
    print(keypoints.xy) # This is the data you will feed into your ML form classifier!