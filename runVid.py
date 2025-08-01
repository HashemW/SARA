import cv2
from ultralytics import YOLO
from tqdm import tqdm # For a helpful progress bar
import os
import sys
    
# --- 1. User-Defined Parameters ---
# Path to your fine-tuned YOLOv8 model
MODEL_PATH = "/fs/nexus-scratch/hwahed/yoloFineTuning/Horse_Keypoints/goodRun!/weights/best.pt"
# Path to the input video
VIDEO_PATH = ''
# Path to save the output video
OUTPUT_DIR = "/fs/nexus-scratch/hwahed/ai_equestrian/testYoloOutputs"

# Confidence threshold for drawing keypoints (e.g., 0.5 means 50% confidence)
CONF_THRESHOLD = 0.5

if len(sys.argv) < 2:
    print("Usage: python coach.py <video_path>")
    sys.exit(1)
VIDEO_PATH = sys.argv[1]
OUTPUT_DIR = os.path.join(OUTPUT_DIR, os.path.basename(VIDEO_PATH).replace('.mp4', '_output.mp4'))
if not os.path.exists(VIDEO_PATH):
    print(f"Video file {VIDEO_PATH} does not exist.")
    sys.exit(1) 
# --- 2. Load Model and Video ---
try:
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video at {VIDEO_PATH}")

    # Get video properties for VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer object to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_DIR, fourcc, fps, (frame_width, frame_height))

    # --- 3. Keypoint Definitions ---
    # Indices for "neck_base": 22, "back_base": 23, "back_end": 24
    # 'neck_base', 'neck_end', 'throat_end', 'back_base', 'back_end'
    # no neck base, neck_end
    # keeping, back_base, throat_end,  back_end
    
    selected_joints_indices = [35]  # Adjusted for your model's keypoints

    # --- 4. Process Video with Streaming ---
    # Use model's streaming feature for memory efficiency
    results_generator = model(VIDEO_PATH, stream=True, verbose=False)

    # Get total frame count for the progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video... Output will be saved to: {OUTPUT_DIR}")
    for results in tqdm(results_generator, total=total_frames):
        annotated_frame = results.orig_img

        if results.keypoints:
            keypoints_data = results.keypoints
            
            # Iterate over each detected object (horse)
            for i in range(len(keypoints_data)):
                # Get the coordinates and confidence scores for the current object
                if keypoints_data.xy[i].shape[0] == 0:
                    print("No keypoints detected for this object.")
                    continue
                coords = keypoints_data.xy[i]
                if coords.shape[0] < 37:
                    print(f"Warning: Expected 37 keypoints, but got {coords.shape[0]}.")
                    continue
                # Get the confidence scores for the current object
                confs = keypoints_data.conf[i]

                # Draw only the selected joints
                for joint_index in range(0, 37):
                    if joint_index < len(coords):
                        x, y = coords[joint_index]
                        confidence = confs[joint_index]

                        # Draw only if confidence is above the threshold
                        if confidence > CONF_THRESHOLD:
                            cv2.circle(annotated_frame, (int(x), int(y)), radius=4, color=(0, 255, 255), thickness=-1)
        
        # Write the annotated frame to the output video file
        out.write(annotated_frame)
        
        # --- Optional: Display frame (can be slow and cause crashes on servers) ---
        # cv2.imshow("Custom Pose Estimation", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

finally:
    # --- 5. Release Resources ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete.")