import os
import sys
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
import math
current_dir = os.path.dirname(os.path.abspath(__file__))

configs_path = os.path.join(current_dir, 'models', 'configs')

src_path = os.path.join(current_dir, 'src')

# Add the folder to sys.path

sys.path.append(configs_path)

sys.path.append(src_path)
# Local Imports
import config as config
from transformer_model import GaitTransformer, calculate_all_features, LABEL_TO_GAIT
from video_processing import (setup_video_writer, draw_rider_guidance, draw_rider_guidance_frontback, 
                              calculate_horse_back_angle)
from inference_landmarks import VirtualLandmarkPredictor
from angle_model import HorsePoseAnalyzer, keypoints_to_bbox_yolo

def run_analysis(video_path):
    """
    Main analysis function.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Initialization ---
    yolo_model = YOLO(config.YOLO_MODEL_PATH)
    
    # Initialize updated Gait Transformer
    gait_model = GaitTransformer(
        num_features=config.RAW_INPUT_SIZE, 
        num_classes=config.NUM_CLASSES,
        d_model=config.D_MODEL,
        nhead=config.N_HEADS,
        num_encoder_layers=config.N_LAYERS,
        dropout=config.DROPOUT_PROB
    ).to(device)
    
    gait_model.load_state_dict(torch.load(config.GAIT_MODEL_PATH, map_location=device))
    gait_model.eval()
    
    analyzer = HorsePoseAnalyzer()
    landmark_model = VirtualLandmarkPredictor()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    ret, frame_0 = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    video_path_no_ext = os.path.basename(os.path.splitext(video_path)[0])
    output_video_path = os.path.join(config.OUTPUT_VIDEO_DIR, f"{video_path_no_ext}_coached.mp4")

    print(f"Output video will be saved to {output_video_path}")
    out, fps = setup_video_writer(cap, output_video_path)

    # --- Buffers ---
    # Updated sequence length (60)
    frame_buffer = deque(maxlen=config.FRAME_SEQUENCE_LENGTH)
    keypoints_buffer = deque(maxlen=3)
    angles_buffer = deque(maxlen=2)
    
    total_back_scores = []
    total_leg_scores = []
    total_arm_scores = []

    # --- Main Loop ---
    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        annotated_frame = current_frame.copy()
        
        # --- 1. YOLO DETECTION ---
        yolo_results = yolo_model(current_frame, verbose=False)

        if not yolo_results or not yolo_results[0].keypoints or len(yolo_results[0].keypoints.xy) == 0:
            out.write(current_frame)
            continue

        result = yolo_results[0]
        full_keypoints_xy = result.keypoints.xy.cpu().numpy()[0]   # (N, 2)
        full_confidences = result.keypoints.conf.cpu().numpy()[0]  # (N,)

        # Create (N, 3) for Analyzer
        full_keypoints_with_conf = np.concatenate(
            (full_keypoints_xy, full_confidences.reshape(-1, 1)), 
            axis=1
        )

        horse_keypoints = full_keypoints_xy[15:]
        horse_bbox = keypoints_to_bbox_yolo(horse_keypoints)
        
        if result.keypoints.xy.shape[1] < 37:
            out.write(current_frame)
            continue

        # --- 2. ORIENTATION ANALYSIS ---
        angle_data = analyzer.analyze_frame(full_keypoints_with_conf, horse_bbox)
        angle_of_orientation = int(angle_data["heading_deg"])
        
        # Create heading vector [cos, sin] for Feature Extraction
        heading_rad = np.radians(angle_of_orientation)
        heading_vector = np.array([np.cos(heading_rad), np.sin(heading_rad)])

        # --- 3. GAIT CLASSIFICATION ---
        last_prediction = None
        normalized_kps = result.keypoints.xyn.cpu().numpy()[0][config.HORSE_LEG_KEYPOINT_INDICES]

        # Check if legs are detected
        if np.all(np.sum(normalized_kps, axis=1) > 0):
            keypoints_buffer.append(normalized_kps)
            
            if len(keypoints_buffer) == 3:
                # Calculate features passing the NEW heading_vector
                features = calculate_all_features(
                    keypoints_buffer[2], 
                    keypoints_buffer[1], 
                    keypoints_buffer[0], 
                    angles_buffer,
                    heading_vector
                )
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
            angles_buffer.clear()

        # --- 4. DRAWING & COACHING ---
        horse_angle, oriented_left = calculate_horse_back_angle(result, None) 

        # Draw Gait Label
        if last_prediction:
            cv2.putText(annotated_frame, f"{last_prediction}", 
                        (int(full_keypoints_xy[config.RIDER_R_EAR_IDX][0]) + 50, 
                         int(full_keypoints_xy[config.RIDER_R_EAR_IDX][1]) - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, config.GREEN, 2)

        # Directional Logic for Rider Guidance
        s_back, s_leg, s_arm = None, None, None
        
        # Side View Right-to-Left
        if 75 <= angle_of_orientation <= 95 and horse_angle is not None:           
            s_back, s_leg, s_arm = draw_rider_guidance(annotated_frame, horse_angle, oriented_left, full_keypoints_xy, full_confidences, angle_of_orientation, config.Direction.RIGHT_TO_LEFT) 
            
        # Side View Left-to-Right
        elif -95 <= angle_of_orientation <= -75 and horse_angle is not None:
            s_back, s_leg, s_arm = draw_rider_guidance(annotated_frame, horse_angle, oriented_left, full_keypoints_xy, full_confidences, angle_of_orientation, config.Direction.LEFT_TO_RIGHT) 
            
        # Back View
        elif -25 <= angle_of_orientation <= 25:  
            landmarks = landmark_model.predict(full_keypoints_with_conf, horse_bbox)
            s_back, s_leg, s_arm = draw_rider_guidance_frontback(annotated_frame, landmarks, full_keypoints_xy, full_confidences, config.Direction.BACK_VIEW)
        
        # Front View
        elif (135 <= angle_of_orientation <= 180) or (-180 <= angle_of_orientation <= -160):  
            landmarks = landmark_model.predict(full_keypoints_with_conf, horse_bbox)
            s_back, s_leg, s_arm = draw_rider_guidance_frontback(annotated_frame, landmarks, full_keypoints_xy, full_confidences, config.Direction.FRONT_VIEW)
            
        # Angled Views
        elif 25 < angle_of_orientation < 75:
            s_back, s_leg, s_arm = draw_rider_guidance(annotated_frame, horse_angle, oriented_left, full_keypoints_xy, full_confidences, angle_of_orientation, config.Direction.RIGHT_TO_LEFT) 
        elif 95 < angle_of_orientation < 135:
            s_back, s_leg, s_arm = draw_rider_guidance(annotated_frame, horse_angle, oriented_left, full_keypoints_xy, full_confidences, angle_of_orientation, config.Direction.TOP_LEFT_TO_BOTTOM_RIGHT) 
        elif -75 < angle_of_orientation < -25:
            s_back, s_leg, s_arm = draw_rider_guidance(annotated_frame, horse_angle, oriented_left, full_keypoints_xy, full_confidences, angle_of_orientation, config.Direction.BOTTOM_RIGHT_TO_TOP_LEFT) 
        else:
            s_back, s_leg, s_arm = draw_rider_guidance(annotated_frame, horse_angle, oriented_left, full_keypoints_xy, full_confidences, angle_of_orientation, config.Direction.TOP_RIGHT_TO_BOTTOM_LEFT) 

        # Accumulate Scores
        if s_back is not None:
            total_back_scores.append(s_back)
            if s_leg is not None: total_leg_scores.append(s_leg)
            if s_arm is not None: total_arm_scores.append(s_arm)

        out.write(annotated_frame)

    # --- Cleanup ---
    print("\n" + "="*40)
    print(" FINAL RIDER ASSESSMENT REPORT")
    print("="*40)

    avg_back = (sum(total_back_scores) / len(total_back_scores)) * 100 if total_back_scores else 0
    avg_leg = (sum(total_leg_scores) / len(total_leg_scores)) * 100 if total_leg_scores else 0
    avg_arm = (sum(total_arm_scores) / len(total_arm_scores)) * 100 if total_arm_scores else 0
    total_avg = (avg_back + avg_leg + avg_arm) / 3 if (total_back_scores and total_leg_scores and total_arm_scores) else 0

    print(f"Position (Back/Torso): {avg_back:.1f}%")
    print(f"Leg Position:          {avg_leg:.1f}%")
    print(f"Arm/Hand Position:     {avg_arm:.1f}%")
    print("-" * 40)
    print(f"OVERALL SCORE:         {total_avg:.1f}%")
    print("="*40 + "\n")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_video_path>")
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(f"Error: Video file {input_path} does not exist.")
        sys.exit(1)

    run_analysis(input_path)