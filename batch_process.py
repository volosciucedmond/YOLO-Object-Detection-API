import os
import csv
import argparse
import cv2
from ultralytics import YOLO
import json
import shutil # directory cleaning

# setup arguments
parser = argparse.ArgumentParser(description="Batch process images")
parser.add_argument("--input", type=str, default="custom_dataset", help="Folder with images")
parser.add_argument("--output", type=str, default="static/batch_results", help="Folder to save results")
parser.add_argument("--classes", type=str, default=None, help="Comma-separated classes (e.g. car,truck)")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

args = parser.parse_args()

# directory cleaning (fresh save)
if os.path.exists(args.output):
    print(f"Cleaning up the old results in '{args.output}'")
    shutil.rmtree(args.output) # deletes the folder and everything in it

# create/recreate the results folder
os.makedirs(args.output)
print(f"Created a new folder: '{args.output}'")

# load model
model = YOLO("yolo26n.pt")

# output preparation
results_data = [] # save json sumamry

# parse filter classes if provided
filter_classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
if filter_classes:
    print(f"Filtering only for {filter_classes}")
    
# run batch
image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpeg', '.jpg'))]
print(f"Found {len(image_files)} images in '{args.input}'")

for img_name in image_files:
    img_path = os.path.join(args.input, img_name)
    
    # run YOLO
    results = model(img_path, conf=args.conf, verbose=False)[0]
    
    original_img = cv2.imread(img_path)
    detections_found = False
    
    image_summary = {"image": img_name, "detections": []}
    
    for box in results.boxes:
        # get the data
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0]) # convert tensor coordinates to int for pixel-based drawing
        
        """ 
        FIlter logic:
        if the user asked for specific classes, skip everything else
        """
        if filter_classes and class_name not in filter_classes:
            continue
        
        # add summary
        image_summary["detections"].append({
            "class": class_name,
            "confidence": round(conf, 2),
            "box": [x1, y1, x2, y2]
        })
        
        # draw box
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(original_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        detections_found = True
        
    # save images only if we found what we were looking for
    if detections_found:
        save_path = os.path.join(args.output, img_name)
        cv2.imwrite(save_path, original_img)
        results_data.append(image_summary)
        print(f"Saved: {img_name}")
    else:
        print(f"Skipped: {img_name} (No relevant object)")
        
# save json
json_path = os.path.join(args.output, "results.json")
with open(json_path, "w") as f:
    json.dump(results_data, f, indent=4)
    
print(f"Done processing {len(image_files)} images")
print(f"Report: {json_path}")

