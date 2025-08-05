import os
import sys
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO, SAM # Import SAM
import math

# Import from our refactored modules
import config
from transformer_model import GaitTransformer, calculate_all_features, LABEL_TO_GAIT
from pose_processing import extract_features_from_keypoints
from video_processing import (setup_video_writer, draw_keypoints, draw_gait_prediction, 
                              draw_rider_guidance, calculate_body_aspect_ratio,
                              calculate_horse_back_angle)

def run_analysis(video_path, segment_rider_flag):
    """
    Main analysis function.
    Args:
        video_path (str): Path to the input video.
        segment_rider_flag (bool): If True, segment the rider.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Initialization ---
    yolo_model = YOLO(config.YOLO_MODEL_PATH)
    gait_model = GaitTransformer(num_features=config.RAW_INPUT_SIZE, num_classes=config.NUM_CLASSES).to(device)
    gait_model.load_state_dict(torch.load(config.GAIT_MODEL_PATH, map_location=device))
    gait_model.eval()

    # NEW: Initialize SAM/YOLO model if segmentation is enabled
    sam_model = None
    person_detector_model = None
    if segment_rider_flag:
        print("Segmentation enabled. Loading SAM model...")
        try:
            sam_model = SAM(config.SAM_MODEL_PATH).to(device)
            person_detector_model = YOLO(config.YOLO_PERSON_MODEL_PATH).to(device)
            print("SAM model loaded successfully.")
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            print("Disabling segmentation for this run.")
            segment_rider_flag = False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    
    video_path_no_ext = os.path.basename(os.path.splitext(video_path)[0])
    output_video_path = os.path.join(config.OUTPUT_VIDEO_DIR, f"{video_path_no_ext}_coached.mp4")
    
    print(f"Output video will be saved to {output_video_path}")
    
    out, fps = setup_video_writer(cap, output_video_path)
    
    # --- Buffers and State Variables ---
    frame_buffer = deque(maxlen=config.FRAME_SEQUENCE_LENGTH)
    keypoints_buffer = deque(maxlen=3)
    angles_buffer = deque(maxlen=2)
    ratio_buffer = deque(maxlen=5) 
    smooth_angle_buffer = deque(maxlen=5)
    
    # --- Main Processing Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- STEP 1: ANALYSIS ON ORIGINAL FRAME ---
        yolo_results = yolo_model(frame, verbose=False)
        
        # If no keypoints, write original frame and skip
        if not yolo_results or not yolo_results[0].keypoints or len(yolo_results[0].keypoints.xy) == 0:
            out.write(frame)
            continue
        
        result = yolo_results[0]
        
        if result.keypoints.xy.shape[1] < 37:
            print("Warning: Not enough keypoints detected.")
            out.write(frame)
            continue
        
        full_keypoints_xy = result.keypoints.xy.cpu().numpy()[0]
        full_confidences = result.keypoints.conf.cpu().numpy()[0]
        
        # --- Gait Classification Logic ---
        last_prediction = None
        normalized_kps = result.keypoints.xyn.cpu().numpy()[0][config.HORSE_LEG_KEYPOINT_INDICES]
        if np.all(np.sum(normalized_kps, axis=1) > 0):
            keypoints_buffer.append(normalized_kps)
            if len(keypoints_buffer) == 3:
                features = calculate_all_features(keypoints_buffer[2], keypoints_buffer[1], keypoints_buffer[0], angles_buffer)
                frame_buffer.append(features)
            if len(frame_buffer) == config.FRAME_SEQUENCE_LENGTH:
                sequence_to_predict = torch.tensor(np.array(frame_buffer), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = gait_model(sequence_to_predict)
                    _, predicted_idx = torch.max(output.data, 1)
                    last_prediction = LABEL_TO_GAIT[predicted_idx.item()]
        else:
            keypoints_buffer.clear()
            frame_buffer.clear()

        # --- Rider Guidance Logic ---
        aspect_ratio, box = calculate_body_aspect_ratio(full_keypoints_xy, full_confidences)
        is_gradeable = False
        smoothed_angle = None
        oriented_left = False
        if aspect_ratio is not None:
            ratio_buffer.append(aspect_ratio)
            smoothed_ratio = np.mean(ratio_buffer)
            if smoothed_ratio < config.MAX_ASPECT_RATIO:
                is_gradeable = True
                angle, oriented_left = calculate_horse_back_angle(result, result.boxes.xywh.cpu().numpy()[0])
                if angle is not None:
                    smooth_angle_buffer.append(angle)
                    smoothed_angle = np.mean(smooth_angle_buffer)

        # --- STEP 2: CREATE THE FRAME TO BE ANNOTATED ---
        # This is where we apply segmentation if the flag is set
        annotated_frame = frame.copy() # Start with a clean copy

        if segment_rider_flag and person_detector_model is not None and sam_model is not None:
            # Use standard YOLO to find the rider
            person_results = person_detector_model(frame, classes=[0], verbose=False) # class 0 is 'person'
            person_boxes = person_results[0].boxes
            
            # Find the most confident person detection
            best_conf = -1; best_box = None
            for i in range(len(person_boxes)):
                if person_boxes.conf[i] > best_conf:
                    best_conf = person_boxes.conf[i]
                    best_box = person_boxes.xyxy[i].cpu().numpy().astype(int)

            # NEW: If no confident person is found, SKIP THIS FRAME from the output video
            if best_conf < config.YOLO_PERSON_CONF_THRESHOLD:
                continue 

            # If a confident person is found, proceed with segmentation
            if best_box is not None:
                sam_results = sam_model(frame, bboxes=best_box, verbose=False)
                if sam_results and sam_results[0].masks is not None:
                    mask_conf = sam_results[0].boxes.conf.cpu().numpy()[0]
                    if mask_conf >= config.SAM_CONF_THRESHOLD:
                        mask = sam_results[0].masks.data[0].cpu().numpy().astype(bool)
                        annotated_frame[mask] = config.BLUE 

        # --- STEP 3: DRAW ALL ANNOTATIONS ON THE (POSSIBLY SEGMENTED) FRAME ---
        if full_confidences[config.NOSE_IDX] > 0.2:
             draw_gait_prediction(annotated_frame, last_prediction, full_keypoints_xy)

        if is_gradeable and smoothed_angle is not None:
            draw_rider_guidance(annotated_frame, smoothed_angle, oriented_left, full_keypoints_xy, full_confidences)
        else:
            # Draw simple posture lines if not in a gradeable position
            _, oriented_left = calculate_horse_back_angle(result, result.boxes.xywh.cpu().numpy()[0])
            if oriented_left:
                points_indices = [config.RIDER_L_SHOULDER_IDX, config.RIDER_L_ANKLE_IDX, config.RIDER_L_ELBOW_IDX, config.RIDER_L_HIP_IDX, config.RIDER_L_WRIST_IDX, config.RIDER_L_KNEE_IDX]
            else:
                points_indices = [config.RIDER_R_SHOULDER_IDX, config.RIDER_R_ANKLE_IDX, config.RIDER_R_ELBOW_IDX, config.RIDER_R_HIP_IDX, config.RIDER_R_WRIST_IDX, config.RIDER_R_KNEE_IDX]
            
            points = [full_keypoints_xy[i] for i in points_indices]
            if all(full_confidences[i] > 0.2 for i in points_indices):
                cv2.line(annotated_frame, tuple(points[0].astype(int)), tuple(points[2].astype(int)), config.LIGHT_ORANGE, 2)
                cv2.line(annotated_frame, tuple(points[2].astype(int)), tuple(points[4].astype(int)), config.LIGHT_ORANGE, 2)
                cv2.line(annotated_frame, tuple(points[3].astype(int)), tuple(points[5].astype(int)), config.LIGHT_ORANGE, 2)
                cv2.line(annotated_frame, tuple(points[5].astype(int)), tuple(points[1].astype(int)), config.LIGHT_ORANGE, 2)

        # --- STEP 4: WRITE THE FINAL FRAME ---
        out.write(annotated_frame)

    # --- Cleanup ---
    print("Finished processing.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Expects: python main.py <input_path> <debug_option> <segment_flag (0 or 1)>
    if len(sys.argv) != 4:
        print("Usage: python main.py <input_video_path> <debug option> <segment_flag>")
        print("  <segment_flag>: 1 to segment the rider, 0 to not segment.")
        sys.exit(1)
        
    input_path = sys.argv[1]
    debug = sys.argv[2]
    
    if debug:
        config.DEBUG = True
    else:
        config.DEBUG = False
        
    try:
        segment_flag = bool(int(sys.argv[3]))
    except ValueError:
        print("Error: segment_flag must be 0 or 1.")
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: Video file {input_path} does not exist.")
        sys.exit(1)
        
    run_analysis(input_path, segment_flag)
