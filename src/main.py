import os
import sys
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
import math
import mediapipe as mp # Import MediaPipe

# Import from our refactored modules
import config
from transformer_model import GaitTransformer, calculate_all_features, LABEL_TO_GAIT
from pose_processing import extract_features_from_keypoints
from video_processing import (setup_video_writer, draw_keypoints, draw_gait_prediction,
                              draw_rider_guidance,
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

    # NEW: Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

    # NOTE: Segmentation logic is temporarily disabled to focus on MediaPipe integration
    # sam_model = None
    # person_detector_model = None
    # if segment_rider_flag:
    #     ...

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    ret, frame_0 = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        exit()

    frame_count = 0
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
    
    # NEW: Buffer for rider angle smoothing (stores angle components)
    rider_angle_buffer = deque(maxlen=10) # Increase maxlen for more smoothness

    # --- Main Processing Loop ---
    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated_frame = current_frame.copy()
        rider_angle = None # Variable to hold the raw angle from MediaPipe
        smoothed_rider_angle = None # Variable to hold the smoothed angle

        # --- STEP 1: YOLO HORSE+RIDER DETECTION ---
        yolo_results = yolo_model(current_frame, verbose=False)

        if not yolo_results or not yolo_results[0].keypoints or len(yolo_results[0].keypoints.xy) == 0:
            out.write(current_frame)
            continue

        result = yolo_results[0]

        if result.keypoints.xy.shape[1] < 37:
            print("Warning: Not enough keypoints detected.")
            out.write(current_frame)
            continue

        # --- NEW: BLAZEPOSE RIDER ANALYSIS ---
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        if boxes.shape[0] > 0:
            box = boxes[0]

            rider_crop = current_frame[box[1]:box[3], box[0]:box[2]]

            if rider_crop.size > 0:
                image_rgb = cv2.cvtColor(rider_crop, cv2.COLOR_BGR2RGB)
                mp_results = pose.process(image_rgb)

                if mp_results.pose_landmarks:
                    landmarks = mp_results.pose_landmarks.landmark

                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                    if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                        dx_3d = left_shoulder.x - right_shoulder.x
                        dz_3d = left_shoulder.z - right_shoulder.z

                        orientation_rad = math.atan2(dz_3d, dx_3d)
                        rider_angle = math.degrees(orientation_rad)

                        # --- NEW: Angle Smoothing Logic ---
                        # Append the (cosine, sine) of the new angle to the buffer
                        rider_angle_buffer.append((math.cos(orientation_rad), math.sin(orientation_rad)))
        
        # Calculate the smoothed angle if the buffer has values
        if rider_angle_buffer:
            # Sum all the cosine and sine components separately
            sum_cos = sum(c for c, s in rider_angle_buffer)
            sum_sin = sum(s for c, s in rider_angle_buffer)
            
            # Calculate the average of the components
            mean_cos = sum_cos / len(rider_angle_buffer)
            mean_sin = sum_sin / len(rider_angle_buffer)
            
            # Convert the averaged components back to a single angle
            smoothed_angle_rad = math.atan2(mean_sin, mean_cos)
            smoothed_rider_angle = math.degrees(smoothed_angle_rad)


        # --- Gait Classification Logic (Unchanged) ---
        last_prediction = None
        full_keypoints_xy = result.keypoints.xy.cpu().numpy()[0]
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

        # UPDATED: Display the smoothed rider angle
        if smoothed_rider_angle is not None:
            # angle_text = f"Rider Angle: {smoothed_rider_angle:.1f}"
            #subtract withers keypoint from top of tail
            # cv2.putText(annotated_frame, angle_text, (50, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
            horse_angle, oriented_left = calculate_horse_back_angle(result, None) # You might need to pass the box here

            # --- DRAW ANNOTATIONS ---
            draw_gait_prediction(annotated_frame, last_prediction, full_keypoints_xy)

            # NEW: Call the guidance function on every frame
            # The function itself will decide whether to draw anything based on the rider's angle
            if horse_angle is not None:
                full_confidences = result.keypoints.conf.cpu().numpy()[0]
                draw_rider_guidance(annotated_frame,
                                    horse_angle,
                                    oriented_left,
                                    full_keypoints_xy,
                                    full_confidences,
                                    rider_orientation_angle=smoothed_rider_angle) # <-- Pass the new angle here

        out.write(annotated_frame)

    # --- Cleanup ---
    print("Finished processing.")
    cap.release()
    out.release()
    pose.close() # Close the MediaPipe pose object
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python main.py <input_video_path> <debug option> <segment_flag>")
        print("  <segment_flag>: 1 to segment the rider, 0 to not segment. (Currently ignored)")
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