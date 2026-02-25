from ultralytics import YOLO
import cv2
import os

# define paths
MODEL_NAME = "yolo26n.pt"
IMAGE_NAME = "test.jpg"

def run_local_test():
    # check if image exists:
    if not os.path.exists(IMAGE_NAME):
        print(f"Error: Image '{IMAGE_NAME}' not found.")
        return
    
    print(f"Loading model '{MODEL_NAME}'...")
    try:
        # load model
        model = YOLO(MODEL_NAME)
        
        # run detection
        print(f"Running detection on '{IMAGE_NAME}'...")
        
        # predict and save the results ("save=True" will save the image with detections in the 
        # specified project folder)
        results = model.predict(IMAGE_NAME, save=True, project = "static/results", name = "test_run")
        
        print(f"Detection completed. Results saved in {results[0].save_dir}")
    
    except Exception as e:
        print(f"Something went wrong: {e}")
        
if __name__ == "__main__":
    run_local_test()
        

        
    