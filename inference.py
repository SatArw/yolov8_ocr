#load button detector
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import os, PIL, io, cv2, torch

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'


nano_path = 'yolov8nano.pt'
s_path = 'yolov8small.pt'
model = YOLO(s_path) #edit the path here to choose your model

#load ocr model
from character_recognition import CharacterRecognizer
recognizer = CharacterRecognizer(verbose=False)

#preprocessing functions
def button_candidates(boxes, scores, image):

    button_scores = []  # stores the score of each button (confidence)
    button_patches = []  # stores the cropped image that encloses the button
    button_positions = []  # stores the coordinates of the bounding box on buttons

    for box, score in zip(boxes, scores):
        if score < 0.5:
            continue

        y_min = int(box[0])
        x_min = int(box[1])
        y_max = int(box[2])
        x_max = int(box[3])

        if x_min < 0 or y_min < 0:
            continue
        button_patch = image[y_min: y_max, x_min: x_max]
        button_patch = cv2.resize(button_patch, (180, 180))

        button_scores.append(score)
        button_patches.append(button_patch)
        button_positions.append([x_min, y_min, x_max, y_max])
    return button_patches, button_positions, button_scores

# Generating a array with paths to test images
test_path = "./test/"
images_arr = []  # Array of image paths
for file_name in os.listdir(test_path):
    images_arr.append(os.path.join(test_path, file_name))

times_det = []
times_lbl = []

for file_path in (images_arr):
    # Button detection
    with open(file_path, 'rb') as f:
        img_np = np.asarray(PIL.Image.open(io.BytesIO(f.read())))
    t0 = cv2.getTickCount()
    preds = model.predict(file_path)
    t1 = cv2.getTickCount()
    time = (t1-t0)/cv2.getTickFrequency()
    times_det.append(time)
    
    for pred in preds:
        boxes = pred.boxes.xyxy.tolist()
        scores = pred.boxes.conf.tolist()
        
    button_patches, button_positions, _ = button_candidates(
        boxes, scores, img_np)
    
    t0 = cv2.getTickCount()
    for button_img in button_patches:
        # get button text and button_score for each of the images in button_patches
        button_text, button_score, _ = recognizer.predict(button_img)
    t1 = cv2.getTickCount()
    time = (t1-t0)/cv2.getTickFrequency()
    times_lbl.append(time)
    
times_total = times_lbl + times_det  # Analyzing total times
arr = np.array(times_total)
# measures of dispersion
avg = np.round(np.mean(arr),5)
min = np.round(np.amin(arr),5)
max = np.round(np.amax(arr),5)
range = np.round(np.ptp(arr),5)
variance = np.round(np.var(arr),5)
sd = np.round(np.std(arr),5)


print("Average = ", avg)
print("Minimum =", min)
print("Maximum =", max)
print("Range =", range)
print("Variance =", variance)
print("Standard Deviation =", sd)
    
    
    
    