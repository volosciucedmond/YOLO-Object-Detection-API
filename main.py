from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import uuid
import os
import cv2

# import the object detector class
from app.detector import ObjectDetector

MODEL_PATH = "yolo26n.pt"
UPLOAD_DIR = "static/results"

# create the API and the static folder
app = FastAPI(
    title="Object Detection API with YOLOv26",
    description="Upload an image and get back the detected objects with bounding boxes and JSON data." 
)

# create the folder where the images are saved
os.makedirs(UPLOAD_DIR, exist_ok=True)

# mount the static folder so users can view the saved images
# if this is not done, we cant acces the saved images via URL
app.mount("/static", StaticFiles(directory="static"), name="static")

# load the model globally
print ("Starting the server and loading YOLO")
try:
    detector = ObjectDetector(model_path=MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    detector = None
    
@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/detect")
def detect_objects(file: UploadFile = File(...)):
    """ Using def instead of async because its not blocking
    the main event loop. Def tells the FastAPI to run this in a separate thread
    """
    
    # safety checks
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        # read data
        image_bytes = file.file.read()
        
        # process
        annotated_img, detections = detector.predict(image_bytes)
        
        if annotated_img is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")
        
        # save results
        filename = f"{uuid.uuid4()}.jpg" # generate a unique filename preventing overwriting
        save_path = os.path.join(UPLOAD_DIR, filename)
        
        # save image using OpenCV
        cv2.imwrite(save_path, annotated_img)
        
        # print to console
        print(f"Detections for {filename}:")
        print(f"{'CLASS':<15} {'CONFIDENCE':<10} {'BOX (x1, y1, x2, y2)'}")
        print("-" * 50)
        
        for d in detections:
            label = d['class']
            conf = f"{d['confidence']:.2f}"
            box = d['bbox']
            print(f"{label:<15} {conf:<10} {box}")
        print("-" * 50 + "\n")
        
        # return JSON
        
        return {
            "success": True,
            "filename": filename,
            "image_url": f"/static/results/{filename}",
            "detections_count": len(detections),
            "detections": detections
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
        