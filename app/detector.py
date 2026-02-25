from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path = "yolo26n.pt"):
        """
        This initialises the YOLO model only once, when the server starts.
        This saves us some seconds on every request.
        """
        print(f"Loading model '{model_path}'")
        self.model = YOLO(model_path)
        
        if self.model is not None:
            print("Model loaded successfully.")
            
    def predict(self, image_bytes: bytes):
        # take the image bytes and convert it to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
            
        # get a standard BGR imgage (OpenCV's default colour format)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        # encoding check
        if img is None:
            return None, None
            
        """
        We want to return only the images that have confidence > 35% 
        Also, the result is a list, so because we send a single image, we take index 0
        """
        results = self.model.predict(img, conf=0.35)
        result = results[0]
            
        # create visual result
        detected_img = result.plot()
            
        # create json output
        detections = []
        for box in result.boxes:
            # we need to convert tensor values to Python types
            detections.append({
                "class": result.names[int(box.cls)],    # gets the class name
                "confidence": float(box.conf),          # gets the confidence score
                "bbox": box.xyxy.tolist()[0]            # gets the positions of the box corners
            })
        return detected_img, detections