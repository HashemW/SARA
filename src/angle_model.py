import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

configs_path = os.path.join(current_dir, 'models/configs') # Then go into folder_a

src_path = os.path.join(current_dir, 'src')

# Add the folder to sys.path

sys.path.append(configs_path)

sys.path.append(src_path)

# Local Imports
import config as config
# --- MODEL CLASSES (Must match training) ---
class OrientationNet(nn.Module):
    def __init__(self, input_dim):
        super(OrientationNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.model(x)

class VectorRegressionNet(nn.Module):
    def __init__(self, input_dim):
        super(VectorRegressionNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x): return self.model(x)

# --- ANALYZER ---
class HorsePoseAnalyzer:
    def __init__(self, angle_model_path=config.ORIENTATION_MODEL_PATH, 
                       vector_model_path=config.VERTICAL_MODEL_PATH):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Heading Model (16 keypoints)
        # 15-26 (Legs), 32,33,35,36 (Spine)
        self.angle_indices = [
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 32, 33, 35, 36
        ]
        
        # 2. Vertical Model (Belly-Back) (12 keypoints - NO HOOVES)
        # Adjust these indices to match your 'CORE_BODYPARTS' mapping in YOLO
        # Spine: Neck(32?), Throat(27?), BackBase(33?), BackEnd(36?)
        # Thighs: 23, 24, 25, 26
        # Knees: 19, 20, 21, 22
        self.vector_indices = [
             32, 27, 33, 36, 
             23, 24, 25, 26,
             19, 20, 21, 22
        ]
        
        # Load Models
        self.angle_model = OrientationNet(3 + len(self.angle_indices)*3).to(self.device)
        self.angle_model.load_state_dict(torch.load(angle_model_path, map_location=self.device))
        self.angle_model.eval()
        
        self.vector_model = VectorRegressionNet(3 + len(self.vector_indices)*3).to(self.device)
        self.vector_model.load_state_dict(torch.load(vector_model_path, map_location=self.device))
        self.vector_model.eval()

    def normalize(self, keypoints, bbox, indices):
        bx, by, bw, bh = bbox
        x_min, y_min = bx - bw/2, by - bh/2
        
        # Bbox features
        feats = [bw, bh, bw/bh if bh>0 else 0]
        
        for idx in indices:
            if idx < len(keypoints):
                # keypoints is (N, 3) -> [x, y, conf]
                kp = keypoints[idx]
                norm_x = (kp[0] - x_min) / bw if bw>0 else 0
                norm_y = (kp[1] - y_min) / bh if bh>0 else 0
                feats.extend([norm_x, norm_y, kp[2]])
            else:
                feats.extend([0.0, 0.0, 0.0])
                
        return torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)

    def analyze_frame(self, keypoints, bbox):
        # 1. Get Forward Vector (Heading)
        input_angle = self.normalize(keypoints, bbox, self.angle_indices)
        with torch.no_grad():
            sincos = self.angle_model(input_angle).cpu().numpy()[0]
        
        # Forward in Camera Space (X, Y, Z)
        # Angle predicts Rotation around Y.
        # If 0 deg = Facing Camera (Z-), then Forward = [sin, 0, -cos] or similar.
        # Based on your plots: Atan2(sin, cos) -> Degrees.
        # Let's assume standard: Forward = [sin, 0, cos]
        vec_forward = np.array([sincos[0], 0, sincos[1]])
        vec_forward = vec_forward / (np.linalg.norm(vec_forward) + 1e-6)

        # 2. Get Up Vector (Belly -> Back)
        input_vec = self.normalize(keypoints, bbox, self.vector_indices)
        with torch.no_grad():
            vec_up = self.vector_model(input_vec).cpu().numpy()[0]
        
        vec_up = vec_up / (np.linalg.norm(vec_up) + 1e-6)

        # 3. CALCULATE RIGHT VECTOR (Geometric Cross Product)
        # Right = Cross(Up, Forward)
        # If Up is roughly [0, 1, 0] and Forward is [0, 0, 1], Right is [1, 0, 0]
        vec_right = np.cross(vec_up, vec_forward)
        vec_right = vec_right / (np.linalg.norm(vec_right) + 1e-6)

        heading_deg = np.degrees(np.arctan2(sincos[0], sincos[1]))

        return {
            "heading_deg": heading_deg,
            "vec_forward": vec_forward,
            "vec_up": vec_up,       # Use this for "Vertical" alignment
            "vec_right": vec_right  # Use this for "Hip-to-Hip" alignment
        }
def keypoints_to_bbox_yolo(keypoints):
    """
    Converts a list of normalized keypoints to a normalized YOLO bounding box format [x_center, y_center, width, height].
    
    Args:
        keypoints (list or np.array): A flat list or array of normalized keypoint coordinates 
                                     [kp1_x, kp1_y, kp1_conf, kp2_x, kp2_y, kp2_conf, ...].
                                     Confidence values (every third element) are ignored for calculation.

    Returns:
        list: Bounding box in normalized YOLO format [x_center, y_center, width, height].
              Returns None if no valid keypoints are provided.
    """
    # Reshape keypoints to a 2D array [num_keypoints, 3] if it's a flat list
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints)
    
    if keypoints.ndim == 1:
        # Assuming format is [x1, y1, conf1, x2, y2, conf2, ...]
        coords = keypoints.reshape(-1, 3)[:, :2]
    else:
        # Assuming format is [x1, y1, conf1] for each row
        coords = keypoints[:, :2]

    # Filter out invalid keypoints if necessary (e.g., those with 0,0 coordinates or very low confidence)
    # This example assumes all provided keypoints are valid.
    
    # Calculate min and max for x and y coordinates
    x_min = np.min(coords[:, 0])
    y_min = np.min(coords[:, 1])
    x_max = np.max(coords[:, 0])
    y_max = np.max(coords[:, 1])

    # Calculate bounding box dimensions in normalized format
    width = x_max - x_min
    height = y_max - y_min
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    return [x_center, y_center, width, height]