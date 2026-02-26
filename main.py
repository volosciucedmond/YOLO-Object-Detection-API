import logging
import os
import uuid
import cv2
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv



# import the object detector class
from app.detector import ObjectDetector

# load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "yolov26.pt")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "static/results")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# logging setup
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("YOLO-API")



# create the API and the static folder
app = FastAPI(
    title="Object Detection API with YOLOv26",
    description="Upload an image and get back the detected objects with bounding boxes and JSON data." 
)

detector = None

# create the folder where the images are saved
os.makedirs(UPLOAD_DIR, exist_ok=True)

# mount the static folder so users can view the saved images
# if this is not done, we cant acces the saved images via URL
app.mount("/static", StaticFiles(directory="static"), name="static")

# startup logic
@app.on_event("startup")
async def startup_event():
    global detector
    logger.info(f"Initialising server. Loading YOLO model from: {MODEL_PATH}")
    try:
        detector = ObjectDetector(model_path=MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load model: {e}", exc_info=True)
        detector = None
        
@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/detect")
def detect_objects(file: UploadFile = File(...)):
    """ Handles the image upload and object detection
    Using 'def' allows FastAPI to run this in an external threadpool.
    """
    if detector is None:
        logger.error("Detection attempt failed: Model is not initialised.")
        raise HTTPException(status_code=500, detail="Model not initialised on server")
    
    if not file.content_type.startswith("image/"):
        logger.warning(f"Rejected upload: Invalid content type {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")
    
    try:
        # read data
        image_bytes = file.file.read()
        
        # process detection
        annotated_img, detections = detector.predict(image_bytes)
        
        if annotated_img is None:
            logger.error("Detector returned None: Possible corrupted image.")
            raise HTTPException(status_code=400, detail="Invalid image file.")
        
        filename = f"{uuid.uuid4()}.jpg"
        save_path = os.path.join(UPLOAD_DIR, filename)
        
        # save results
        cv2.imwrite(save_path, annotated_img)
        
        logger.info(f"File processed: {filename} | Detectiosn found: {len(detections)}")
        for d in detections:
            logger.debug(f"Detected: {d['class']} ({d['confidence']:.2f}) at {d['bbox']}")
        
        return {"success": True,
                "filename": filename,
                "image_url": f"/static/results/{filename}",
                "detections_count": len(detections),
                "detections": detections
        } 
    
    except Exception as e:
        logger.error(f"Error during detection process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during processing.")
    
if __name__ == "__main__":
    # Host and port from env for easier Docker integration
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8001))
    uvicorn.run(app, host=host, port=port)
        