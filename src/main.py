import os
import sys
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
import math

# Import from our refactored modules
import config
from transformer_model import load_transformer_model
from pose_processing import extract_features_from_keypoints
from video_processing import (setup_video_writer, draw_keypoints, draw_gait_prediction, 
                              draw_rider_guidance, calculate_body_aspect_ratio,
                              calculate_horse_back_angle)

def run_analysis(video_path):
    # --- Initialization ---
    yolo_model = YOLO(config.YOLO_MODEL_PATH)
    transformer_model = load_transformer_model(config.TRANSFORMER_MODEL_PATH)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    video_file_name = os.path.basename(video_path)
    video_path_no_ext = os.path.splitext(video_file_name)[0]
    output_video_path = os.path.join(config.OUTPUT_VIDEO_DIR, f"{video_path_no_ext}_coached.mp4")
    print(f"Output video will be saved to {output_video_path}")
    
    out, fps = setup_video_writer(cap, output_video_path)
    
    # --- Buffers and State Variables ---
    frame_buffer = deque(maxlen=config.FRAME_SEQUENCE_LENGTH)
    keypoints_buffer = deque(maxlen=3)
    prediction_history = deque(maxlen=int(fps / 2))
    ratio_buffer = deque(maxlen=5) # Buffer for smoothing the aspect ratio
    angle_buffer = deque(maxlen=5) # Buffer for smoothing the guidance angle
    
    # --- Main Processing Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        yolo_results = yolo_model(frame, verbose=False)
        
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
            
        # --- Gait Classification ---
        normalized_kps = result.keypoints.xyn.cpu().numpy()[0][config.HORSE_LEG_KEYPOINT_INDICES]
        if np.all(np.sum(normalized_kps, axis=1) > 0):
            keypoints_buffer.append(normalized_kps)
            if len(keypoints_buffer) == 3:
                features = extract_features_from_keypoints(keypoints_buffer)
                frame_buffer.append(features)
            if len(frame_buffer) == config.FRAME_SEQUENCE_LENGTH:
                sequence_np = np.array(frame_buffer, dtype=np.float32)
                sequence_tensor = torch.from_numpy(sequence_np).unsqueeze(0).to(config.DEVICE)
                with torch.no_grad():
                    output = transformer_model(sequence_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    top3_probs, top3_indices = torch.topk(probabilities, 3)
                    prediction_history.append(list(zip(top3_indices[0].cpu().numpy(), top3_probs[0].cpu().numpy())))
        else:
            keypoints_buffer.clear()
            frame_buffer.clear()

        # --- Visualization ---
        # draw_keypoints(frame, full_keypoints_xy)
        if full_confidences[config.NOSE_IDX] > 0.2:
             draw_gait_prediction(frame, prediction_history, full_keypoints_xy)

        # --- Aspect Ratio for Grading Condition ---
        aspect_ratio, box = calculate_body_aspect_ratio(full_keypoints_xy, full_confidences)
        
        if aspect_ratio is not None:
            ratio_buffer.append(aspect_ratio)
            smoothed_ratio = np.mean(ratio_buffer)
            if config.DEBUG:
                # --- Visualization of the bounding box and ratio ---
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2) # Cyan box
                ratio_text = f"Ratio: {smoothed_ratio:.2f}"
                cv2.putText(frame, ratio_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            box_xywh = result.boxes.xywh.cpu().numpy()[0]
            angle, oriented_left = calculate_horse_back_angle(result, box_xywh)
            # --- Check if horse is in a gradeable position ---
            if smoothed_ratio < config.MAX_ASPECT_RATIO:
                # If the view is good, calculate the back angle for the guidance viz
                
                
                if angle is not None:
                    angle_buffer.append(angle)
                    smoothed_angle = np.mean(angle_buffer)
                    draw_rider_guidance(frame, smoothed_angle, oriented_left, full_keypoints_xy, full_confidences)
            else:
                # If not in a good position, inform the user
                if config.DEBUG:
                    cv2.putText(frame, "Reposition for Grading", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 3, cv2.LINE_AA)
                # draw circles on elbow, shoulder, wrist, hip, ankle, and knee
                if oriented_left:
                    points = [full_keypoints_xy[config.RIDER_L_SHOULDER_IDX], 
                                   full_keypoints_xy[config.RIDER_L_ANKLE_IDX], 
                                   full_keypoints_xy[config.RIDER_L_ELBOW_IDX], 
                                   full_keypoints_xy[config.RIDER_L_HIP_IDX], 
                                   full_keypoints_xy[config.RIDER_L_WRIST_IDX], 
                                   full_keypoints_xy[config.RIDER_L_KNEE_IDX]]
                    for point in points:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 1, config.ORANGE, -1)
                    cv2.line(frame, tuple(points[0].astype(int)), tuple(points[2].astype(int)), config.LIGHT_ORANGE, 2)
                    cv2.line(frame, tuple(points[2].astype(int)), tuple(points[4].astype(int)), config.LIGHT_ORANGE, 2)
                    cv2.line(frame, tuple(points[3].astype(int)), tuple(points[5].astype(int)), config.LIGHT_ORANGE, 2)
                    cv2.line(frame, tuple(points[5].astype(int)), tuple(points[1].astype(int)), config.LIGHT_ORANGE, 2)
                    
                else:
                    points = [full_keypoints_xy[config.RIDER_R_SHOULDER_IDX], 
                                   full_keypoints_xy[config.RIDER_R_ANKLE_IDX], 
                                   full_keypoints_xy[config.RIDER_R_ELBOW_IDX], 
                                   full_keypoints_xy[config.RIDER_R_HIP_IDX], 
                                   full_keypoints_xy[config.RIDER_R_WRIST_IDX], 
                                   full_keypoints_xy[config.RIDER_R_KNEE_IDX]]
                    for point in points:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 1, config.ORANGE, -1)
                    cv2.line(frame, tuple(points[0].astype(int)), tuple(points[2].astype(int)), config.LIGHT_ORANGE, 2)
                    cv2.line(frame, tuple(points[2].astype(int)), tuple(points[4].astype(int)), config.LIGHT_ORANGE, 2)
                    cv2.line(frame, tuple(points[3].astype(int)), tuple(points[5].astype(int)), config.LIGHT_ORANGE, 2)
                    cv2.line(frame, tuple(points[5].astype(int)), tuple(points[1].astype(int)), config.LIGHT_ORANGE, 2)
        
        out.write(frame)

    # --- Cleanup ---
    print("Finished processing.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python main.py <video_path> DEBUG")
        
        sys.exit(1)
        
    video_path = sys.argv[1]
    config.DEBUG = int(sys.argv[2])
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        sys.exit(1)
        
    run_analysis(video_path)
