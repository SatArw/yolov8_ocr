from ultralytics import YOLO
import torch 

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
    
model = YOLO('yolov8nano.pt')
results = model.predict('/home/satarw/yolonas/test/1640_jpg.rf.647f67b253ed71d42fcadd2170f7ab59.jpg')

for res in results:
    print(res.boxes.xyxy)
    print(res.boxes.conf)