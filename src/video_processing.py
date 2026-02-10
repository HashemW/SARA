import cv2
import math
import numpy as np
import os
import sys
import config
from config import (RIDER_L_HIP_IDX, RIDER_R_HIP_IDX, CLASSES,
                    RIDER_L_SHOULDER_IDX, RIDER_R_SHOULDER_IDX, GREEN, RED,
                    LEFT_FRONT_SHOULDER_IDX, RIGHT_FRONT_SHOULDER_IDX, TOP_OF_TAIL_IDX,
                    HORSE_BODY_AND_LEGS_INDICES, ANGLE_CONSTANT, RIDER_L_KNEE_IDX,
                    RIDER_R_KNEE_IDX, RIDER_L_ANKLE_IDX, RIDER_R_ANKLE_IDX,
                    LEFT_WITHERS_IDX, LIGHT_GREEN, YELLOW)

# Perspective correction constants are no longer needed
# PERSPECTIVE_CORRECTION_FACTOR = 0.04
# NEUTRAL_ANGLE = 88.0

def rotate_to_target_angle(
    fixed_point: tuple[float, float],
    pivot_point: tuple[float, float],
    point_to_rotate: tuple[float, float],
    target_angle_degrees: float,
    clockwise: bool = False
) -> tuple[float, float]:
    """
    Rotates a point around a pivot to achieve a target angle relative to a fixed point.
    """
    px, py = pivot_point
    fx, fy = fixed_point
    rx, ry = point_to_rotate

    fixed_translated = (fx - px, fy - py)
    point_to_rotate_translated = (rx - px, ry - py)
    reference_angle_rad = math.atan2(fixed_translated[1], fixed_translated[0])
    radius = math.sqrt(point_to_rotate_translated[0]**2 + point_to_rotate_translated[1]**2)

    if radius == 0:
        return pivot_point

    target_angle_rad = math.radians(target_angle_degrees)
    if clockwise:
        target_angle_rad = -target_angle_rad

    new_angle_rad = reference_angle_rad + target_angle_rad
    new_x_translated = radius * math.cos(new_angle_rad)
    new_y_translated = radius * math.sin(new_angle_rad)
    new_x = new_x_translated + px
    new_y = new_y_translated + py

    return (new_x, new_y)

def calculate_horse_back_angle(result, box_xywh):
    """
    Calculates the angle of the horse's back from withers to tail.
    """
    keypoints_xy = result.keypoints.xy.cpu().numpy()[0]
    confidences = result.keypoints.conf.cpu().numpy()[0]

    if confidences[LEFT_WITHERS_IDX] < 0.2 or confidences[TOP_OF_TAIL_IDX] < 0.2:
        return None, None

    withers_x, withers_y = keypoints_xy[LEFT_WITHERS_IDX]
    tail_x, tail_y = keypoints_xy[TOP_OF_TAIL_IDX]

    oriented_left = withers_x < tail_x
    dx = withers_x - tail_x
    dy = withers_y - tail_y
    angle = math.degrees(math.atan2(dy, dx))

    return angle, oriented_left

def setup_video_writer(cap, output_path):
    """Initializes and returns a VideoWriter object."""
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return out, fps

def draw_keypoints(frame, pixel_keypoints):
    """Draws detected keypoints on the frame."""
    for i, (x, y) in enumerate(pixel_keypoints):
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 255), -1)

def draw_gait_prediction(frame, prediction_history, keypoints_xy):
    """Draws the top gait prediction on the frame."""
    nose_x, nose_y = keypoints_xy[config.NOSE_IDX]
    text = f"{prediction_history}"
    cv2.putText(frame, text, (int(nose_x + 50), int(nose_y - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 3, cv2.LINE_AA)

def draw_rider_guidance(frame, horse_back_angle, oriented_left, full_keypoints_with_conf, 
                        rider_confidences, angle_of_orientation, direction):
    """
    Analyzes and visualizes rider posture with corrections for 3D perspective.
    
    Returns:
        tuple: (score_back, score_leg, score_arm)
               Scores are floats (0.0 to 1.0) or None if detection failed.
    """
    
    # Extract XY coordinates from the [x, y, conf] array
    keypoints_xy = full_keypoints_with_conf[:, :2] 
    
    # -------------------------------------------------------------------------
    # 1. INDICES & DIRECTION SETUP
    # -------------------------------------------------------------------------
    if oriented_left:
        # Horse is traveling Left (Head on Left, Tail on Right)
        hip_idx, shoulder_idx = config.RIDER_L_HIP_IDX, config.RIDER_L_SHOULDER_IDX
        knee_idx, ankle_idx = config.RIDER_L_KNEE_IDX, config.RIDER_L_ANKLE_IDX
        wrist_id, elbow_id = config.RIDER_L_WRIST_IDX, config.RIDER_L_ELBOW_IDX
        
        # SIDE SIGN: Determines which way the "barrel shift" pushes the leg.
        # If looking at the Left side, a turn towards camera pushes the leg Left.
        side_sign = -1 
    else:
        # Horse is traveling Right
        hip_idx, shoulder_idx = config.RIDER_R_HIP_IDX, config.RIDER_R_SHOULDER_IDX
        knee_idx, ankle_idx = config.RIDER_R_KNEE_IDX, config.RIDER_R_ANKLE_IDX
        wrist_id, elbow_id = config.RIDER_R_WRIST_IDX, config.RIDER_R_ELBOW_IDX
        
        # If looking at Right side, a turn towards camera pushes the leg Right.
        side_sign = 1

    # -------------------------------------------------------------------------
    # 2. CONFIDENCE CHECK
    # -------------------------------------------------------------------------
    required_indices = [hip_idx, shoulder_idx, knee_idx, ankle_idx, elbow_id, wrist_id]
    
    # If any keypoint is (0,0), it's likely undetected. Return None scores.
    if np.any(keypoints_xy[required_indices] == 0):
        return None, None, None

    # Get Actual Coordinates (The "Real" Rider)
    H = np.array(keypoints_xy[hip_idx])      # Hip
    S = np.array(keypoints_xy[shoulder_idx]) # Shoulder
    K = np.array(keypoints_xy[knee_idx])     # Knee
    A = np.array(keypoints_xy[ankle_idx])    # Ankle
    E = np.array(keypoints_xy[elbow_id])     # Elbow
    W = np.array(keypoints_xy[wrist_id])     # Wrist

    # -------------------------------------------------------------------------
    # 3. VIRTUAL 3D PROJECTION MATH
    # -------------------------------------------------------------------------
    # We must convert the "Idea" of a posture into 2D screen coordinates.
    # angle_of_orientation: -90 or 90 is pure Side View. 0 or 180 is Front/Back.
    rad_angle = math.radians(angle_of_orientation)

    # A. SQUASH FACTOR (Depth Perception)
    # How "wide" should the leg look?
    # - At 90 deg (Side View), sin(90) = 1.0 -> We see the full leg width.
    # - At 0 deg (Front View), sin(0) = 0.0  -> The leg width is squashed to a line.
    projection_factor = abs(math.sin(rad_angle))
    
    # B. BARREL SHIFT FACTOR (Ribcage Width)
    # The rider's leg isn't a flat sticker; it wraps around the horse's barrel.
    # As the horse turns, the leg moves physically Left or Right on the screen.
    # - At 90 deg (Side), cos(90) = 0 -> No shift (leg is centered on hip).
    # - At 45 deg, cos(45) = 0.7 -> Leg shifts "outward" relative to the hip.
    barrel_shift_factor = math.cos(rad_angle) * side_sign

    # -------------------------------------------------------------------------
    # 4. LEG LOGIC (GRAVITY + BARREL OFFSET)
    # -------------------------------------------------------------------------
    
    # STEP A: Ideal Shoulder (Gravity Aligned)
    # In proper riding, the torso should be vertical, regardless of horse tilt.
    # We project the shoulder DIRECTLY UP from the hip.
    len_hs = np.linalg.norm(S - H)
    S_ideal = np.array([H[0], H[1] - len_hs]) 

    # STEP B: Ideal Ankle (Barrel Offset)
    len_ha = np.linalg.norm(A - H)
    
    # Estimate the "Outward" shift caused by the horse's ribcage.
    # We assume the barrel pushes the leg out by approx 20% of the leg length.
    MAX_BARREL_WIDTH = len_ha * 0.20 
    
    # Calculate the exact pixel shift for this specific viewing angle
    current_barrel_offset_x = MAX_BARREL_WIDTH * barrel_shift_factor
    
    # The Ideal Ankle is vertical (gravity) + the calculated barrel shift
    A_ideal = np.array([H[0] + current_barrel_offset_x, H[1] + len_ha]) 

    # STEP C: Ideal Knee (Triangle Geometry)
    # We construct a virtual triangle: Hip -> Knee -> Ankle
    len_thigh = np.linalg.norm(K - H)
    len_shin = np.linalg.norm(A - K)
    
    # Distance between Hip and our calculated Ideal Ankle
    d = np.linalg.norm(A_ideal - H)
    
    # Clamp distance to ensure the triangle is mathematically possible
    d = min(d, len_thigh + len_shin - 0.1) 
    d = max(d, abs(len_thigh - len_shin) + 0.1)

    # Calculate 'a' (vertical distance down the leg line to the knee perpendicular)
    a = (len_thigh**2 - len_shin**2 + d**2) / (2 * d)
    
    # Calculate 'h_leg' (how far the knee sticks out in a perfect side view)
    h_leg = math.sqrt(max(0, len_thigh**2 - a**2)) 
    
    # SQUASH: Multiply the protrusion by our projection factor (sin angle).
    # If looking from the front, the knee protrusion disappears.
    knee_x_protrusion = h_leg * projection_factor
    if oriented_left: knee_x_protrusion = -knee_x_protrusion 
    
    # Project the knee point along the vector from Hip to Ideal Ankle
    vec_H_A = A_ideal - H
    vec_H_A_norm = vec_H_A / (np.linalg.norm(vec_H_A) + 1e-6)
    K_base = H + vec_H_A_norm * a
    
    # Final Ideal Knee Coordinate
    K_ideal = np.array([K_base[0] + knee_x_protrusion, K_base[1]])

    # -------------------------------------------------------------------------
    # 5. ARM LOGIC (ELBOW-TO-BIT LINE)
    # -------------------------------------------------------------------------
    
    # STEP A: Ideal Elbow (Neutral Seat)
    # The upper arm should hang vertically from the shoulder.
    len_upper_arm = np.linalg.norm(E - S)
    E_ideal = np.array([S[0], S[1] + len_upper_arm])

    # STEP B: Ideal Wrist (Dynamic Bit Targeting)
    len_forearm = np.linalg.norm(W - E)
    M = keypoints_xy[config.MOUTH_IDX]
    
    # Check if we successfully detected the horse's mouth
    if M[0] > 0 and M[1] > 0:
        # Dynamic: The straight line from Elbow to Horse's Mouth
        vec_E_M = M - E_ideal
        dist_E_M = np.linalg.norm(vec_E_M)
        
        if dist_E_M > 0:
            vec_E_M_norm = vec_E_M / dist_E_M
            # The ideal wrist lies on this line, at forearm-length distance
            W_ideal = E_ideal + vec_E_M_norm * len_forearm
        else:
            W_ideal = np.array([E_ideal[0], E_ideal[1] + len_forearm])
    else:
        # Fallback: If mouth is hidden, assume a standard 60-degree forward carry
        ideal_forearm_angle_deg = 60 
        # Calculate Y (down) and X (forward) components
        wrist_dy = len_forearm * math.cos(math.radians(ideal_forearm_angle_deg))
        # Important: Apply projection_factor to the X component (squashing forward reach)
        wrist_dx = len_forearm * math.sin(math.radians(ideal_forearm_angle_deg)) * projection_factor
        
        if oriented_left:
            W_ideal = np.array([E_ideal[0] - wrist_dx, E_ideal[1] + wrist_dy])
        else:
            W_ideal = np.array([E_ideal[0] + wrist_dx, E_ideal[1] + wrist_dy])

    # -------------------------------------------------------------------------
    # 6. SCORING & VISUALIZATION
    # -------------------------------------------------------------------------
    # We use Cosine Similarity: Dot Product of vectors / Product of magnitudes.
    # 1.0 = Perfect alignment, -1.0 = Opposite direction.
    
    score_back = 0.0
    score_leg = 0.0
    score_arm = 0.0
    
    color_back, color_leg, color_arm = RED, RED, RED
    
    # --- Back Score (Torso Alignment) ---
    V_actual_back = S - H
    V_ideal_back = S_ideal - H
    if np.linalg.norm(V_actual_back) > 0:
        # Simple Dot Product Similarity
        score_back = np.dot(V_actual_back, V_ideal_back) / (np.linalg.norm(V_actual_back) * np.linalg.norm(V_ideal_back))
        # Clamp to 0-1 for safety (though cos sim is -1 to 1)
        score_back = max(0.0, score_back)
        
        if score_back > 0.98: color_back = LIGHT_GREEN
        elif score_back > 0.90: color_back = YELLOW

    # --- Leg Score (Thigh + Shin Alignment) ---
    # We compare the combined vector of [Hip->Knee, Knee->Ankle]
    V_actual_leg = np.concatenate((K - H, A - K))
    V_ideal_leg = np.concatenate((K_ideal - H, A_ideal - K_ideal))
    if np.linalg.norm(V_actual_leg) > 0:
        score_leg = np.dot(V_actual_leg, V_ideal_leg) / (np.linalg.norm(V_actual_leg) * np.linalg.norm(V_ideal_leg))
        score_leg = max(0.0, score_leg)
        
        if score_leg > 0.96: color_leg = LIGHT_GREEN
        elif score_leg > 0.88: color_leg = YELLOW
        
    # --- Arm Score (Upper Arm + Forearm Alignment) ---
    V_actual_arm = np.concatenate((E - S, W - E))
    V_ideal_arm = np.concatenate((E_ideal - S, W_ideal - E_ideal))
    if np.linalg.norm(V_actual_arm) > 0:
        score_arm = np.dot(V_actual_arm, V_ideal_arm) / (np.linalg.norm(V_actual_arm) * np.linalg.norm(V_ideal_arm))
        score_arm = max(0.0, score_arm)
        
        if score_arm > 0.95: color_arm = LIGHT_GREEN
        elif score_arm > 0.88: color_arm = YELLOW

    # --- Draw Actual Lines (Colored by Score) ---
    cv2.line(frame, tuple(H.astype(int)), tuple(S.astype(int)), color_back, 3)
    cv2.line(frame, tuple(H.astype(int)), tuple(K.astype(int)), color_leg, 3)
    cv2.line(frame, tuple(K.astype(int)), tuple(A.astype(int)), color_leg, 3)
    
    cv2.line(frame, tuple(S.astype(int)), tuple(E.astype(int)), color_arm, 3)
    cv2.line(frame, tuple(E.astype(int)), tuple(W.astype(int)), color_arm, 3)

    # --- Draw Ideal Ghost Lines (Green Reference) ---
    cv2.line(frame, tuple(S_ideal.astype(int)), tuple(H.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)
    cv2.line(frame, tuple(H.astype(int)), tuple(K_ideal.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)
    cv2.line(frame, tuple(K_ideal.astype(int)), tuple(A_ideal.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)
    
    cv2.line(frame, tuple(S.astype(int)), tuple(E_ideal.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)
    cv2.line(frame, tuple(E_ideal.astype(int)), tuple(W_ideal.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)

    # Return the raw scores for averaging in main.py
    return score_back, score_leg, score_arm
        

def draw_rider_guidance_frontback(frame, landmarks, keypoints_xy, confidences, direction):
    """
    Analyzes rider posture from Front/Back views.
    
    Returns:
        tuple: (score_back, score_leg, score_arm)
    """

    # -------------------------------------------------------------------------
    # 1. SETUP & SAFETY CHECKS
    # -------------------------------------------------------------------------
    left_hand = keypoints_xy[config.RIDER_L_WRIST_IDX]
    right_hand = keypoints_xy[config.RIDER_R_WRIST_IDX]
    left_handc = confidences[config.RIDER_L_WRIST_IDX]
    right_handc = confidences[config.RIDER_R_WRIST_IDX]
    
    # Initialize Colors & Scores
    color_arm = RED
    color_back = RED
    score_back = 0.0
    score_arm = 0.0
    score_leg = 0.0 

    # Convert Landmarks
    p_left = np.array(landmarks['left'])
    p_right = np.array(landmarks['right'])
    
    # Calculate Rider Mid-Hip 
    l_hip = keypoints_xy[config.RIDER_L_HIP_IDX]
    r_hip = keypoints_xy[config.RIDER_R_HIP_IDX]
    mid_hip = (l_hip + r_hip) / 2.0

    # -------------------------------------------------------------------------
    # 2. TORSO ALIGNMENT (The "Centering" Logic)
    # -------------------------------------------------------------------------
    # (Torso logic remains unchanged)
    
    vec_rib_line = p_right - p_left
    vec_hip_rel = mid_hip - p_left
    line_len_sq = np.dot(vec_rib_line, vec_rib_line)
    
    if line_len_sq > 0:
        t = np.dot(vec_hip_rel, vec_rib_line) / line_len_sq
        
        # Only draw if Hip is roughly centered between ribs
        if 0.0 <= t <= 1.0:
            
            # --- CALCULATE IDEAL "UP" VECTOR ---
            perp_vec = np.array([-vec_rib_line[1], vec_rib_line[0]])
            norm = np.linalg.norm(perp_vec)
            perp_vec = perp_vec / (norm + 1e-6)

            # --- ORIENTATION CHECK ---
            l_shoulder = keypoints_xy[config.RIDER_L_SHOULDER_IDX]
            r_shoulder = keypoints_xy[config.RIDER_R_SHOULDER_IDX]
            mid_shoulder = (l_shoulder + r_shoulder) / 2.0
            
            rider_torso_vec = mid_shoulder - mid_hip
            
            if np.dot(perp_vec, rider_torso_vec) < 0:
                perp_vec = -perp_vec
            
            # --- DRAWING TORSO ---
            torso_len = np.linalg.norm(rider_torso_vec)
            line_len = torso_len * 1.5 
            
            end_pt = (
                int(mid_hip[0] + perp_vec[0] * line_len),
                int(mid_hip[1] + perp_vec[1] * line_len)
            )
            
            ideal_back_vec = end_pt - mid_hip
            actual_back_vec = mid_shoulder - mid_hip
            
            # Score Back
            score_back = np.dot(actual_back_vec, ideal_back_vec) / (np.linalg.norm(actual_back_vec) * np.linalg.norm(ideal_back_vec))
            score_back = abs(score_back)
            
            if score_back >= 0.98: color_back = LIGHT_GREEN
            elif score_back >= 0.90: color_back = YELLOW
                
            cv2.line(frame, (int(mid_hip[0]), int(mid_hip[1])), end_pt, config.GREEN, 4)
            cv2.line(frame, (int(mid_hip[0]), int(mid_hip[1])), (int(mid_shoulder[0]), int(mid_shoulder[1])), color_back, 4)

            # -------------------------------------------------------------------------
            # 3. ARM ALIGNMENT (Hand Levelness) - MODIFIED
            # -------------------------------------------------------------------------
            # Checks: 
            # 1. High Confidence (> 0.7)
            # 2. Front View Only
            # 3. Spatial Check: Hands must be "Outside" the ears (Wider than the head)
            
            if (right_handc > 0.7 and left_handc > 0.7 and direction == config.Direction.FRONT_VIEW):
                
                # Get Horse Ears
                h_left_ear = keypoints_xy[config.LEFT_EAR_IDX]
                h_right_ear = keypoints_xy[config.RIGHT_EAR_IDX]
                
                # --- SPATIAL FILTER: HANDS WIDER THAN EARS ---
                # In Front View:
                # - Screen Left side: Rider Right Hand (Low X) vs Horse Right Ear (Low X)
                # - Screen Right side: Rider Left Hand (High X) vs Horse Left Ear (High X)
                
                # "Left arm more to the left" -> Rider Right Wrist (Screen Left) < Horse Right Ear
                # "Right arm more to the right" -> Rider Left Wrist (Screen Right) > Horse Left Ear
                
                is_hands_wide = (right_hand[0] < h_right_ear[0]) and (left_hand[0] > h_left_ear[0])

                if is_hands_wide:
                    # Proceed with drawing
                    
                    norm_rib = np.linalg.norm(vec_rib_line)
                    rib_dir_vec = vec_rib_line / (norm_rib + 1e-6)
                    
                    mid_hand = (left_hand + right_hand) / 2
                    point_hand = left_hand - right_hand
                    hand_spread_len = np.linalg.norm(point_hand)
                    draw_len = hand_spread_len / 2

                    end_pt_1 = (
                        int(mid_hand[0] + rib_dir_vec[0] * draw_len),
                        int(mid_hand[1] + rib_dir_vec[1] * draw_len)
                    )

                    end_pt_2 = (
                        int(mid_hand[0] - rib_dir_vec[0] * draw_len),
                        int(mid_hand[1] - rib_dir_vec[1] * draw_len)
                    )

                    # Score Arms
                    ideal_arm_vec = np.array([end_pt_1[0] - end_pt_2[0], end_pt_1[1] - end_pt_2[1]])
                    actual_arm_vec = left_hand - right_hand
                    
                    cos_sim = np.dot(actual_arm_vec, ideal_arm_vec) / (np.linalg.norm(actual_arm_vec) * np.linalg.norm(ideal_arm_vec))
                    score_arm = abs(cos_sim)
                    
                    if score_arm >= 0.98: color_arm = LIGHT_GREEN
                    elif score_arm >= 0.90: color_arm = YELLOW
                    
                    cv2.line(frame, end_pt_1, end_pt_2, config.GREEN, 4)
                    cv2.line(frame, (int(left_hand[0]), int(left_hand[1])), (int(right_hand[0]), int(right_hand[1])), color_arm, 4)

    # -------------------------------------------------------------------------
    # 4. RETURN SCORES
    # -------------------------------------------------------------------------
    return score_back, score_leg, score_arm