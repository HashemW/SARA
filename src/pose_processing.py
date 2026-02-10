import numpy as np
import math
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

configs_path = os.path.join(current_dir, 'models/configs') # Then go into folder_a

src_path = os.path.join(current_dir, 'src')

# Add the folder to sys.path

sys.path.append(configs_path)

sys.path.append(src_path)

# Local Imports
import config as config
def euclidean_dist(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt(np.sum((p1 - p2)**2))

def extract_features_from_keypoints(keypoints_buffer):
    """
    Extracts a feature vector from a sequence of keypoints.
    The buffer should contain [p_prev_prev, p_prev, p_curr].
    """
    p_prev_prev, p_prev, p_curr = keypoints_buffer
    
    # Feature 1: Bounding box of current keypoints
    x_coords, y_coords = p_curr[:, 0], p_curr[:, 1]
    features = [np.max(x_coords) - np.min(x_coords), np.max(y_coords) - np.min(y_coords)]
    
    # Feature 2: Flattened current keypoints
    features.extend(p_curr.flatten())
    
    # Feature 3: Velocities
    velocities = p_curr - p_prev
    features.extend(velocities.flatten())
    
    # Feature 4: Accelerations
    prev_velocities = p_prev - p_prev_prev
    accelerations = velocities - prev_velocities
    features.extend(accelerations.flatten())
    
    # Feature 5: Inter-hoof distances
    features.append(euclidean_dist(p_curr[KP_MAP['L_Front_Hoof']], p_curr[KP_MAP['R_Front_Hoof']]))
    features.append(euclidean_dist(p_curr[KP_MAP['L_Back_Hoof']], p_curr[KP_MAP['R_Back_Hoof']]))
    features.append(euclidean_dist(p_curr[KP_MAP['L_Front_Hoof']], p_curr[KP_MAP['L_Back_Hoof']]))
    features.append(euclidean_dist(p_curr[KP_MAP['R_Front_Hoof']], p_curr[KP_MAP['R_Back_Hoof']]))
    
    # Feature 6: Vertical movement of centroid
    features.append(np.mean(p_curr[:, 1]) - np.mean(p_prev[:, 1]))
    
    current_angles = np.array([
        calculate_angle(p_curr[KP_MAP['L_Front_Shoulder']], p_curr[KP_MAP['L_Front_Knee']], p_curr[KP_MAP['L_Front_Hoof']]),
        calculate_angle(p_curr[KP_MAP['R_Front_Shoulder']], p_curr[KP_MAP['R_Front_Knee']], p_curr[KP_MAP['R_Front_Hoof']]),
        calculate_angle(p_curr[KP_MAP['L_Back_Stifle']], p_curr[KP_MAP['L_Back_Knee']], p_curr[KP_MAP['L_Back_Hoof']]),
        calculate_angle(p_curr[KP_MAP['R_Back_Stifle']], p_curr[KP_MAP['R_Back_Knee']], p_curr[KP_MAP['R_Back_Hoof']]),
    ])
    features.extend(current_angles)
    angles_buffer.append(current_angles)

    if len(angles_buffer) < 2:
        angular_velocities = np.zeros(4)
    else:
        angular_velocities = current_angles - angles_buffer[0]
    features.extend(angular_velocities)
    hoof_features = calculate_hoof_features(current_keypoints)
    features.extend(hoof_features)
    frame_buffer.append(features)
    return features

