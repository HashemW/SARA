import cv2
import math
import numpy as np
from config import (RIDER_L_HIP_IDX, RIDER_R_HIP_IDX, CLASSES, 
                    RIDER_L_SHOULDER_IDX, RIDER_R_SHOULDER_IDX, GREEN, RED,
                    LEFT_FRONT_SHOULDER_IDX, RIGHT_FRONT_SHOULDER_IDX, TOP_OF_TAIL_IDX,
                    HORSE_BODY_AND_LEGS_INDICES, ANGLE_CONSTANT, RIDER_L_KNEE_IDX,
                    RIDER_R_KNEE_IDX, RIDER_L_ANKLE_IDX, RIDER_R_ANKLE_IDX,
                    LEFT_WITHERS_IDX, LIGHT_GREEN, YELLOW)
import config

def rotate_to_target_angle(
    fixed_point: tuple[float, float],
    pivot_point: tuple[float, float],
    point_to_rotate: tuple[float, float],
    target_angle_degrees: float,
    clockwise: bool = False
) -> tuple[float, float]:
    """
    Rotates a point around a pivot to achieve a target angle relative to a fixed point.

    The function calculates the new coordinates for `point_to_rotate` such that the angle
    formed by `fixed_point -> pivot_point -> new_point` is equal to `target_angle_degrees`.

    Args:
        fixed_point: A tuple (x, y) representing the static reference point.
        pivot_point: A tuple (x, y) representing the center of rotation.
        point_to_rotate: A tuple (x, y) representing the point to be moved.
        target_angle_degrees: The desired angle in degrees.
        clockwise: If True, the rotation will be in the clockwise direction. 
                     Defaults to False (counter-clockwise).

    Returns:
        A tuple (x, y) for the new position of the rotated point.
    """
    # Unpack coordinates for clarity
    px, py = pivot_point
    fx, fy = fixed_point
    rx, ry = point_to_rotate

    # --- Step 1: Translate points so the pivot is at the origin (0,0) ---
    fixed_translated = (fx - px, fy - py)
    point_to_rotate_translated = (rx - px, ry - py)

    # --- Step 2: Calculate the angle of the fixed point's vector ---
    reference_angle_rad = math.atan2(fixed_translated[1], fixed_translated[0])

    # --- Step 3: Calculate the distance (radius) from the pivot to the point to rotate ---
    radius = math.sqrt(point_to_rotate_translated[0]**2 + point_to_rotate_translated[1]**2)
    
    if radius == 0:
        return pivot_point

    # --- Step 4: Calculate the new angle ---
    target_angle_rad = math.radians(target_angle_degrees)
    
    if clockwise:
        target_angle_rad = -target_angle_rad

    new_angle_rad = reference_angle_rad + target_angle_rad

    # --- Step 5: Calculate the new coordinates for the rotated point (still translated) ---
    new_x_translated = radius * math.cos(new_angle_rad)
    new_y_translated = radius * math.sin(new_angle_rad)

    # --- Step 6: Translate the new point back to the original coordinate system ---
    new_x = new_x_translated + px
    new_y = new_y_translated + py

    return (new_x, new_y)

def calculate_body_aspect_ratio(keypoints_xy, confidences):
    """
    Calculates the aspect ratio of a bounding box around the horse's body and legs.
    This is a stable way to detect foreshortening.

    Returns:
        - ratio (float): The height/width ratio of the bounding box.
        - box (tuple): The (x, y, w, h) of the bounding box for visualization.
    """
    points = []
    for i in HORSE_BODY_AND_LEGS_INDICES:
        if confidences[i] > 0.2:
            points.append(keypoints_xy[i])
    
    if len(points) < 4:
        return None, None

    points = np.array(points)
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    
    if w == 0:
        return None, None
        
    aspect_ratio = h / w
    
    return aspect_ratio, (x, y, w, h)


def calculate_horse_back_angle(result, box_xywh):
    """
    Calculates the angle of the horse's back from withers to tail.
    This is used for the rider guidance visualization.
    Returns the angle in degrees and whether the horse is oriented left.
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


def draw_rider_guidance(frame, angle, oriented_left, keypoints_xy, confidences):
    """
    Draws the ideal rider posture and color-codes the actual posture based on similarity.
    """
    # --- 1. SELECT KEYPOINTS BASED ON ORIENTATION ---
    if oriented_left:
        angle_with_constant = angle + ANGLE_CONSTANT
        hip_idx = RIDER_L_HIP_IDX
        shoulder_idx = RIDER_L_SHOULDER_IDX
        knee_idx = RIDER_L_KNEE_IDX
        ankle_idx = RIDER_L_ANKLE_IDX
        wrist_id = config.RIDER_L_WRIST_IDX
        elbow_id = config.RIDER_L_ELBOW_IDX
        clockwise_status = [True, False]
    else:
        angle_with_constant = angle - ANGLE_CONSTANT
        hip_idx = RIDER_R_HIP_IDX
        shoulder_idx = RIDER_R_SHOULDER_IDX
        knee_idx = RIDER_R_KNEE_IDX
        ankle_idx = RIDER_R_ANKLE_IDX
        wrist_id = config.RIDER_R_WRIST_IDX
        elbow_id = config.RIDER_R_ELBOW_IDX
        clockwise_status = [False, True]
        
    # --- 2. CHECK CONFIDENCE AND GET KEYPOINT COORDINATES ---
    required_indices = [hip_idx, shoulder_idx, knee_idx, ankle_idx, elbow_id, wrist_id]
    if all(confidences[i] > 0.2 for i in required_indices):
        H = np.array(keypoints_xy[hip_idx])
        S = np.array(keypoints_xy[shoulder_idx])
        K = np.array(keypoints_xy[knee_idx])
        A = np.array(keypoints_xy[ankle_idx])
        E = np.array(keypoints_xy[elbow_id])
        W = np.array(keypoints_xy[wrist_id])
        
        # --- 3. DEFINE THE IDEAL STRAIGHT LINE (SHOULDER-HIP-ANKLE) ---
        V_up_ideal_dir = np.array([math.cos(math.radians(angle_with_constant)), math.sin(math.radians(angle_with_constant))])
        len_hs = np.linalg.norm(S - H)
        S_ideal = H + V_up_ideal_dir * len_hs
        V_down_ideal_dir = -V_up_ideal_dir
        len_ha = np.linalg.norm(A - H)
        A_ideal = H + V_down_ideal_dir * len_ha

        # --- 4. CALCULATE IDEAL KNEE POSITION ---
        len_thigh = np.linalg.norm(K - H)
        len_shin = np.linalg.norm(A - K)
        d = np.linalg.norm(A_ideal - H)
        K_ideal = None
        if d <= len_thigh + len_shin and d >= abs(len_thigh - len_shin):
            a = (len_thigh**2 - len_shin**2 + d**2) / (2 * d)
            h = math.sqrt(max(0, len_thigh**2 - a**2))
            P2 = H + a * (A_ideal - H) / d
            K_ideal_1 = np.array([P2[0] + h * (A_ideal[1] - H[1]) / d, P2[1] - h * (A_ideal[0] - H[0]) / d])
            K_ideal_2 = np.array([P2[0] - h * (A_ideal[1] - H[1]) / d, P2[1] + h * (A_ideal[0] - H[0]) / d])
            original_bend_sign = np.sign(np.cross(K - H, A - H))
            if original_bend_sign == 0: K_ideal = K_ideal_1
            elif np.sign(np.cross(K_ideal_1 - H, A_ideal - H)) == original_bend_sign: K_ideal = K_ideal_1
            else: K_ideal = K_ideal_2
        
        # --- 5. SCORE AND COLOR THE POSTURES (BACK & LEG) ---
        color_back = RED
        color_leg = RED

        # Score back posture
        V_actual_back = S - H
        V_ideal_back = S_ideal - H
        if np.linalg.norm(V_actual_back) > 0 and np.linalg.norm(V_ideal_back) > 0:
            similarity_back = np.dot(V_actual_back, V_ideal_back) / (np.linalg.norm(V_actual_back) * np.linalg.norm(V_ideal_back))
            if similarity_back > 0.99: color_back = LIGHT_GREEN
            elif similarity_back > 0.95: color_back = YELLOW

        # Score leg posture
        if K_ideal is not None:
            V_actual_leg = np.concatenate((K - H, A - K))
            V_ideal_leg = np.concatenate((K_ideal - H, A_ideal - K_ideal))
            if np.linalg.norm(V_actual_leg) > 0 and np.linalg.norm(V_ideal_leg) > 0:
                similarity_leg = np.dot(V_actual_leg, V_ideal_leg) / (np.linalg.norm(V_actual_leg) * np.linalg.norm(V_ideal_leg))
                if similarity_leg > 0.99: color_leg = LIGHT_GREEN
                elif similarity_leg > 0.95: color_leg = YELLOW

        # --- 6. DRAW THE POSTURES (BACK & LEG) ---
        cv2.line(frame, tuple(H.astype(int)), tuple(S.astype(int)), color_back, 2)
        cv2.line(frame, tuple(H.astype(int)), tuple(K.astype(int)), color_leg, 2)
        cv2.line(frame, tuple(K.astype(int)), tuple(A.astype(int)), color_leg, 2)

        if K_ideal is not None:
            cv2.line(frame, tuple(S_ideal.astype(int)), tuple(H.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)
            cv2.line(frame, tuple(H.astype(int)), tuple(K_ideal.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)
            cv2.line(frame, tuple(K_ideal.astype(int)), tuple(A_ideal.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)
            
        # --- 7. ARM POSTURE GUIDANCE & SCORING ---
        # Calculate ideal arm position based on target angles
        E_ideal = np.array(rotate_to_target_angle(H, S, E, 15, clockwise=clockwise_status[1]))
        W_ideal = np.array(rotate_to_target_angle(S, E_ideal, W, 140, clockwise=clockwise_status[0]))
        
        # Score arm posture using cosine similarity
        color_arm = RED
        V_actual_arm = np.concatenate((E - S, W - E))
        V_ideal_arm = np.concatenate((E_ideal - S, W_ideal - E_ideal))

        if np.linalg.norm(V_actual_arm) > 0 and np.linalg.norm(V_ideal_arm) > 0:
            similarity_arm = np.dot(V_actual_arm, V_ideal_arm) / (np.linalg.norm(V_actual_arm) * np.linalg.norm(V_ideal_arm))
            if similarity_arm > 0.98:
                color_arm = LIGHT_GREEN
            elif similarity_arm > 0.90:
                color_arm = YELLOW
                
        # --- 8. DRAW ARM POSTURE ---
        # Draw the actual arm with color-coded feedback
        cv2.line(frame, tuple(S.astype(int)), tuple(E.astype(int)), color_arm, 2)
        cv2.line(frame, tuple(E.astype(int)), tuple(W.astype(int)), color_arm, 2)

        # Draw the ideal arm posture in Green for reference
        cv2.line(frame, tuple(S_ideal.astype(int)), tuple(E_ideal.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)
        cv2.line(frame, tuple(E_ideal.astype(int)), tuple(W_ideal.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)

        
        

