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

# NEW: Constants for tuning the perspective correction of the ideal heel position
PERSPECTIVE_CORRECTION_FACTOR = 0.04  # Controls how much the heel shifts per degree of angle deviation.
NEUTRAL_ANGLE = 88.0                  # The 'perfectly sideways' angle where heel is directly under the hip.

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

def draw_rider_guidance(frame, angle, oriented_left, keypoints_xy, confidences, rider_orientation_angle=None):
    """
    Draws the ideal rider posture, accounting for 3D perspective, but only if the rider is sideways to the camera.
    """
    # --- 1. CHECK IF THE RIDER IS IN A GRADEABLE ORIENTATION ---
    if rider_orientation_angle is None:
        return

    grade_this_frame = False
    if not oriented_left:
        if 80 < rider_orientation_angle < 100: grade_this_frame = True
    else:
        if -100 < rider_orientation_angle < -80: grade_this_frame = True

    if not grade_this_frame:
        return

    # --- 2. SELECT KEYPOINTS BASED ON ORIENTATION ---
    if oriented_left:
        angle_with_constant = angle + ANGLE_CONSTANT
        hip_idx, shoulder_idx = RIDER_L_HIP_IDX, RIDER_L_SHOULDER_IDX
        knee_idx, ankle_idx = RIDER_L_KNEE_IDX, RIDER_L_ANKLE_IDX
        wrist_id, elbow_id = config.RIDER_L_WRIST_IDX, config.RIDER_L_ELBOW_IDX
        clockwise_status = [True, False]
    else:
        angle_with_constant = angle - ANGLE_CONSTANT
        hip_idx, shoulder_idx = RIDER_R_HIP_IDX, RIDER_R_SHOULDER_IDX
        knee_idx, ankle_idx = RIDER_R_KNEE_IDX, RIDER_R_ANKLE_IDX
        wrist_id, elbow_id = config.RIDER_R_WRIST_IDX, config.RIDER_R_ELBOW_IDX
        clockwise_status = [False, True]

    # --- 3. CHECK CONFIDENCE AND GET KEYPOINT COORDINATES ---
    required_indices = [hip_idx, shoulder_idx, knee_idx, ankle_idx, elbow_id, wrist_id]
    if all(confidences[i] > 0.2 for i in required_indices):
        H, S = np.array(keypoints_xy[hip_idx]), np.array(keypoints_xy[shoulder_idx])
        K, A = np.array(keypoints_xy[knee_idx]), np.array(keypoints_xy[ankle_idx])
        E, W = np.array(keypoints_xy[elbow_id]), np.array(keypoints_xy[wrist_id])

        # --- 4. DEFINE THE IDEAL STRAIGHT LINE (SHOULDER-HIP-ANKLE) ---
        V_up_ideal_dir = np.array([math.cos(math.radians(angle_with_constant)), math.sin(math.radians(angle_with_constant))])
        len_hs = np.linalg.norm(S - H)
        S_ideal = H + V_up_ideal_dir * len_hs
        V_down_ideal_dir = -V_up_ideal_dir
        len_ha = np.linalg.norm(A - H)
        A_ideal = H + V_down_ideal_dir * len_ha

        # --- 5. NEW: APPLY PERSPECTIVE CORRECTION TO IDEAL HEEL ---
        if len_hs > 0:
            # Get a direction vector that is horizontal/perpendicular to the ideal posture
            H_dir = np.array([-V_up_ideal_dir[1], V_up_ideal_dir[0]])

            # Calculate deviation from the 'perfectly sideways' angle
            if not oriented_left:  # Moving right (positive angle)
                angle_diff = rider_orientation_angle - NEUTRAL_ANGLE
            else:  # Moving left (negative angle)
                angle_diff = rider_orientation_angle - (-NEUTRAL_ANGLE)

            # Calculate the horizontal shift based on deviation, a factor, and the rider's size
            offset_distance = angle_diff * PERSPECTIVE_CORRECTION_FACTOR * len_hs

            # Apply the calculated offset to the ideal ankle position
            A_ideal = A_ideal + (H_dir * offset_distance)

        # --- 6. CALCULATE IDEAL KNEE POSITION (using the new A_ideal) ---
        len_thigh, len_shin = np.linalg.norm(K - H), np.linalg.norm(A - K)
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

        # --- 7. SCORE AND COLOR THE POSTURES (BACK & LEG) ---
        color_back, color_leg = RED, RED
        V_actual_back, V_ideal_back = S - H, S_ideal - H
        if np.linalg.norm(V_actual_back) > 0 and np.linalg.norm(V_ideal_back) > 0:
            similarity_back = np.dot(V_actual_back, V_ideal_back) / (np.linalg.norm(V_actual_back) * np.linalg.norm(V_ideal_back))
            if similarity_back > 0.99: color_back = LIGHT_GREEN
            elif similarity_back > 0.95: color_back = YELLOW

        if K_ideal is not None:
            V_actual_leg, V_ideal_leg = np.concatenate((K - H, A - K)), np.concatenate((K_ideal - H, A_ideal - K_ideal))
            if np.linalg.norm(V_actual_leg) > 0 and np.linalg.norm(V_ideal_leg) > 0:
                similarity_leg = np.dot(V_actual_leg, V_ideal_leg) / (np.linalg.norm(V_actual_leg) * np.linalg.norm(V_ideal_leg))
                if similarity_leg > 0.99: color_leg = LIGHT_GREEN
                elif similarity_leg > 0.95: color_leg = YELLOW

        # --- 8. DRAW THE POSTURES (BACK & LEG) ---
        cv2.line(frame, tuple(H.astype(int)), tuple(S.astype(int)), color_back, 2)
        cv2.line(frame, tuple(H.astype(int)), tuple(K.astype(int)), color_leg, 2)
        cv2.line(frame, tuple(K.astype(int)), tuple(A.astype(int)), color_leg, 2)

        if K_ideal is not None:
            cv2.line(frame, tuple(S_ideal.astype(int)), tuple(H.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)
            cv2.line(frame, tuple(H.astype(int)), tuple(K_ideal.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)
            cv2.line(frame, tuple(K_ideal.astype(int)), tuple(A_ideal.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)

        # --- 9. ARM POSTURE GUIDANCE & SCORING ---
        E_ideal = np.array(rotate_to_target_angle(H, S, E, 15, clockwise=clockwise_status[1]))
        W_ideal = np.array(rotate_to_target_angle(S, E_ideal, W, 140, clockwise=clockwise_status[0]))
        color_arm = RED
        V_actual_arm, V_ideal_arm = np.concatenate((E - S, W - E)), np.concatenate((E_ideal - S, W_ideal - E_ideal))
        if np.linalg.norm(V_actual_arm) > 0 and np.linalg.norm(V_ideal_arm) > 0:
            similarity_arm = np.dot(V_actual_arm, V_ideal_arm) / (np.linalg.norm(V_actual_arm) * np.linalg.norm(V_ideal_arm))
            if similarity_arm > 0.98: color_arm = LIGHT_GREEN
            elif similarity_arm > 0.90: color_arm = YELLOW

        # --- 10. DRAW ARM POSTURE ---
        cv2.line(frame, tuple(S.astype(int)), tuple(E.astype(int)), color_arm, 2)
        cv2.line(frame, tuple(E.astype(int)), tuple(W.astype(int)), color_arm, 2)
        cv2.line(frame, tuple(S_ideal.astype(int)), tuple(E_ideal.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)
        cv2.line(frame, tuple(E_ideal.astype(int)), tuple(W_ideal.astype(int)), GREEN, 2, lineType=cv2.LINE_AA)