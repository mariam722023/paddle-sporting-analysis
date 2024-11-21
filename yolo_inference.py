from ultralytics import YOLO

model= YOLO("models/best (1).pt")
result = model.predict('input_videos/input_video.mp4', save = True)

print(result[0])


for box in result[0].boxes:
     print(box)