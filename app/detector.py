import os
import logging
import cv2
import numpy as np
from ultralytics import YOLO

# get the logger defined in main.py
logger = logging.getLogger("YOLO-API.detector")

class ObjectDetector:
    def __init__(self, model_path: str = "yolo26n.pt"):
        """Initialises the YOLO model once during server startup to optimise perforamnce"""
        
        logger.info (f"Attempting to load YOLO model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            # default confidence threshold from environment or fallback to 0.35
            self.default_conf = float(os.getenv("CONFIDENCE_THRESHOLD", 0.35))
            logger.info(f"YOLO model loaded successfully with confidence threshold: {self.default_conf}")
        except Exception as e:
            logger.error(f"Failed to initialise YOLO model: {e}", exc_info=True)
            raise
        
    def predict(self, image_bytes: bytes):
        """ Processes image bytes, performs detection and returns annotated image and data. """
        try:
            # convert image bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # decode to standard bgr image
            img = cv2.imdecode(nparr, cv2. IMREAD_COLOR)
            
            if img is None:
                logger.warning("Image decoding failed: Received invalid bytes or unsupported format.")
                return None, None
            
            # perform prediction using the configurable threshold
            results = self.model.predict(img, conf=self.default_conf)
            result = results[0]
            
            # generated the visual representation (plotted bounding boxes)
            detected_img = result.plot()
            
            # extract detection data into a clean JSON-ready format
            detections = []
            for box in result.boxes: 
                detections.append({
                    "class": result.names[int(box.cls)],
                    "confidence": round(float(box.conf), 4),
                    "bbox": [round(coord, 2) for coord in box.xyxy.tolist()[0]]
                })
                
            logger.debug(f"Detection successful. Found {len(detections)} objects.")
            return detected_img, detections
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            return None, None