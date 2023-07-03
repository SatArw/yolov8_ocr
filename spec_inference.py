#load button detector
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import os, PIL, io, cv2, torch
import time, subprocess

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:   
    map_location='cpu'


nano_path = 'yolov8nano.pt'
s_path = 'yolov8small.pt'
model = YOLO(nano_path) #edit the path here to choose your model

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
test_path = "./test_images/"
images_arr = []  # Array of image paths
for file_name in os.listdir(test_path):
    images_arr.append(os.path.join(test_path, file_name))

# Measure initial CPU, memory, and disk usage
initial_cpu_usage = subprocess.check_output("top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}'", shell=True)
initial_cpu_usage = float(initial_cpu_usage.decode().strip())
initial_memory_info = subprocess.check_output("free -m | grep Mem", shell=True)
initial_memory_info = initial_memory_info.decode().split()
initial_used_memory = int(initial_memory_info[2])

t0 = time.time()
for file_path in (images_arr):
    # Button detection
    with open(file_path, 'rb') as f:
        img_np = np.asarray(PIL.Image.open(io.BytesIO(f.read())))
    preds = model.predict(file_path)
    
    for pred in preds:
        boxes = pred.boxes.xyxy.tolist()
        scores = pred.boxes.conf.tolist()
        
    button_patches, button_positions, _ = button_candidates(
        boxes, scores, img_np)

    for button_img in button_patches:
        # get button text and button_score for each of the images in button_patches
        button_text, button_score, _ = recognizer.predict(button_img)

t1 = time.time()

# Measure final CPU, memory, and disk usage
final_cpu_usage = subprocess.check_output("top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}'", shell=True)
final_cpu_usage = float(final_cpu_usage.decode().strip())

final_memory_info = subprocess.check_output("free -m | grep Mem", shell=True)
final_memory_info = final_memory_info.decode().split()
final_used_memory = int(final_memory_info[2])

# Calculate and print the usage differences
cpu_usage_diff = final_cpu_usage - initial_cpu_usage
memory_used_diff = final_used_memory - initial_used_memory
    
print(f"Time elapsed = {t1-t0}s")
print(f"CPU usage = {cpu_usage_diff}%")
print(f"Memory used = {memory_used_diff}MB")

    