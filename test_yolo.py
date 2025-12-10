from ultralytics import YOLO

print("Loading YOLOv8 pose model...")
model = YOLO('yolov8n-pose.pt')

print("Testing on sample image...")
results = model('https://ultralytics.com/images/bus.jpg')

print(f"Success! Detected {len(results[0].keypoints)} people")
print("YOLOv8 is working correctly!")
