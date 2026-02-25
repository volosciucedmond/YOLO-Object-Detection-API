import cv2
import os
import argparse

def extract_frames(video_path, output_folder, target_images=30):
    """ 
    extracts a specific number of frames from a video
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: couldn't open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // target_images)
    
    count = 0
    saved_count = 0
    
    print(f"Processing {video_path} ({total_frames} frames)...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:03d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            if saved_count >= target_images:
                break
            
        count += 1
    
    cap.release()
    print(f"Extracted {saved_count} images to '{output_folder}'")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="custom_dataset", help="Output folder")
    parser.add_argument("--count", type=int, default=30, help="Number of frames to extract")
    args = parser.parse_args()
    
    extract_frames(args.video, args.output, args.count)
     