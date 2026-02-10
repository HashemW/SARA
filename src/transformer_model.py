import torch
import torch.nn as nn
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

# --- HELPER FUNCTIONS ---
def calculate_angle(p1, p2, p3):
    v1 = p1 - p2; v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0: return 180.0
    angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    return np.degrees(angle)

def euclidean_dist(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# --- FEATURE EXTRACTION (THE LEAN 22) ---
def calculate_all_features(p_curr, p_prev, p_prev_prev, angles_buffer, heading_vector):
    """
    Calculates the 22 features for the Lean Model.
    Args:
        p_curr: Current normalized keypoints (N, 2)
        p_prev: Previous normalized keypoints (N, 2)
        p_prev_prev: (Unused in Lean model, kept for compatibility)
        angles_buffer: deque for angular velocity
        heading_vector: np.array([cos_angle, sin_angle])
    """
    features = []

    # indices based on KP_MAP
    # 0:LF_Hoof, 1:RF_Hoof, 2:LB_Hoof, 3:RB_Hoof
    # 4:LF_Knee, 5:RF_Knee, 6:LB_Knee, 7:RB_Knee
    # 8:LF_Shoulder, 9:RF_Shoulder, 10:LB_Stifle, 11:RB_Stifle

    # 1. LEG ANGLES (4 Features)
    # LF, RF, LB, RB
    current_angles = np.array([
        calculate_angle(p_curr[8], p_curr[4], p_curr[0]),
        calculate_angle(p_curr[9], p_curr[5], p_curr[1]),
        calculate_angle(p_curr[10], p_curr[6], p_curr[2]),
        calculate_angle(p_curr[11], p_curr[7], p_curr[3])
    ])
    features.extend(current_angles)
    
    # 2. ANGULAR VELOCITY (4 Features)
    # We use angles_buffer to track changes
    if len(angles_buffer) > 0:
        prev_angles = angles_buffer[-1]
        ang_vel = current_angles - prev_angles
    else:
        ang_vel = np.zeros(4)
    
    features.extend(ang_vel)
    angles_buffer.append(current_angles)

    # 3. HOOF PAIR DISTANCES (6 Features)
    features.append(euclidean_dist(p_curr[0], p_curr[1])) # LF-RF (Front Width)
    features.append(euclidean_dist(p_curr[2], p_curr[3])) # LB-RB (Back Width)
    features.append(euclidean_dist(p_curr[0], p_curr[2])) # LF-LB (Left Lateral)
    features.append(euclidean_dist(p_curr[1], p_curr[3])) # RF-RB (Right Lateral)
    features.append(euclidean_dist(p_curr[0], p_curr[3])) # LF-RB (Diagonal 1)
    features.append(euclidean_dist(p_curr[1], p_curr[2])) # RF-LB (Diagonal 2)

    # 4. HOOF HEIGHTS (5 Features)
    # Relative heights + Grounded count
    hooves = p_curr[[0, 1, 2, 3]]
    min_y = np.min(hooves[:, 1])
    relative_heights = hooves[:, 1] - min_y
    features.extend(relative_heights) # 4 features
    
    threshold = 0.03
    hooves_grounded = np.sum(relative_heights < threshold)
    features.append(hooves_grounded) # 1 feature

    # 5. VERTICAL OSCILLATION (1 Feature)
    features.append(np.mean(p_curr[:, 1]) - np.mean(p_prev[:, 1]))

    # 6. LEARNED ORIENTATION (2 Features)
    features.extend(heading_vector)

    # TOTAL: 4 + 4 + 6 + 5 + 1 + 2 = 22 Features
    return np.array(features)

# --- MODEL ARCHITECTURE ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Shape (1, max_len, d_model) for batch_first=True
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice dim 1 (sequence length)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class GaitTransformer(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 d_model: int = 64,      # MATCHES YOUR TRAINING LOG (2.7M params)
                 nhead: int = 2,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.model_type = 'Transformer'
        
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.d_model = d_model
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        # Global Average Pooling
        output = output.mean(dim=1)
        
        output = self.classifier(output)
        return output

LABEL_TO_GAIT = {0: 'Walking', 1: 'Trotting', 2: 'Cantering', 3: 'Gallop', 4: 'Standing'}


# --- Main Visualization Logic ---
# import cv2
# import torch
# import numpy as np
# import torch.nn as nn
# from ultralytics import YOLO
# from collections import deque
# import math

# YOLO_MODEL_PATH = "C:/Users/hashe/ai_equestrian/models/best.pt" 
# GAIT_MODEL_PATH = "C:/Users/hashe/ai_equestrian/models/96.97accuracy.pth"
# INPUT_VIDEO_PATH = "C:/Users/hashe/ai_equestrian/input_videos/dressagewestern.mp4"
# OUTPUT_VIDEO_PATH = "output_video.mp4"

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)
    
# class GaitTransformer(nn.Module):
#     def __init__(self,
#                  num_features: int,
#                  num_classes: int,
#                  d_model: int = 128,
#                  nhead: int = 8,
#                  num_encoder_layers: int = 3,
#                  dim_feedforward: int = 512,
#                  dropout: float = 0.1):
#         super().__init__()
#         self.model_type = 'Transformer'
        
#         # This linear layer projects your 87 features into the model's dimension (d_model)
#         self.input_projection = nn.Linear(num_features, d_model)
        
#         self.pos_encoder = PositionalEncoding(d_model, dropout)
        
#         encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
#         self.d_model = d_model
        
#         # Classification head
#         self.classifier = nn.Linear(d_model, num_classes)
        
#     def forward(self, src: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             src: Tensor, shape [batch_size, seq_len, num_features]
#         """
#         # Project input features to model dimension
#         src = self.input_projection(src) * math.sqrt(self.d_model)
#         src = self.pos_encoder(src)
        
#         # Pass through the transformer encoder
#         output = self.transformer_encoder(src)
        
#         # We use the output of the first token ([CLS] token style) for classification
#         output = output[:, 0, :]
        
#         # Final classification
#         output = self.classifier(output)
#         return output
    
# SEQUENCE_LENGTH = 20
# NUM_FEATURES = 87
# NUM_CLASSES = 5
# HORSE_LEG_KEYPOINT_INDICES = list(range(15, 27))
# LABEL_TO_GAIT = {0: 'Walking', 1: 'Trotting', 2: 'Cantering', 3: 'Gallop', 4: 'Standing'}
# KP_MAP = {
#     'L_Front_Hoof': 0, 'R_Front_Hoof': 1, 'L_Back_Hoof': 2, 'R_Back_Hoof': 3,
#     'L_Front_Knee': 4, 'R_Front_Knee': 5, 'L_Back_Knee': 6, 'R_Back_Knee': 7,
#     'L_Front_Shoulder': 8, 'R_Front_Shoulder': 9,
#     'R_Back_Stifle': 10, 'L_Back_Stifle': 11
# }

# # --- Copy your feature calculation functions here ---
# # It's crucial that these are IDENTICAL to the ones used for training
# def calculate_angle(p1, p2, p3):
#     v1 = p1 - p2; v2 = p3 - p2
#     dot_product = np.dot(v1, v2); norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
#     if norm_product == 0: return 180.0
#     angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
#     return np.degrees(angle)

# # ... include all other feature functions like calculate_orientation_features, euclidean_dist etc. ...
# # For now, let's assume a placeholder for the full feature calculation
# def euclidean_dist(p1, p2):
#     return np.sqrt(np.sum((p1 - p2)**2))

# # Ensure calculate_angle is also present, which it already is.

# def calculate_all_features(p_curr, p_prev, p_prev_prev, angles_buffer):
#     """
#     Calculates the 87-feature vector, matching the training script EXACTLY.
#     """
#     features = []

#     # 1. Base Features (Coordinates, Velocity, Acceleration)
#     x_coords, y_coords = p_curr[:, 0], p_curr[:, 1]
#     features.extend([np.max(x_coords) - np.min(x_coords), np.max(y_coords) - np.min(y_coords)])
#     features.extend(p_curr.flatten())
#     velocities = p_curr - p_prev
#     features.extend(velocities.flatten())
#     prev_velocities = p_prev - p_prev_prev
#     features.extend((velocities - prev_velocities).flatten())

#     # 2. Distance & Oscillation Features
#     features.append(euclidean_dist(p_curr[KP_MAP['L_Front_Hoof']], p_curr[KP_MAP['R_Front_Hoof']]))
#     features.append(euclidean_dist(p_curr[KP_MAP['L_Back_Hoof']], p_curr[KP_MAP['R_Back_Hoof']]))
#     features.append(euclidean_dist(p_curr[KP_MAP['L_Front_Hoof']], p_curr[KP_MAP['L_Back_Hoof']]))
#     features.append(euclidean_dist(p_curr[KP_MAP['R_Front_Hoof']], p_curr[KP_MAP['R_Back_Hoof']]))
#     features.append(np.mean(p_curr[:, 1]) - np.mean(p_prev[:, 1]))

#     # 3. Orientation Features (COMMENTED OUT to match the 87-feature model)
#     # orientation_features = calculate_orientation_features(p_curr)
#     # features.extend(orientation_features)

#     # 4. Angle and Angular Velocity Features
#     current_angles = np.array([
#         calculate_angle(p_curr[KP_MAP['L_Front_Shoulder']], p_curr[KP_MAP['L_Front_Knee']], p_curr[KP_MAP['L_Front_Hoof']]),
#         calculate_angle(p_curr[KP_MAP['R_Front_Shoulder']], p_curr[KP_MAP['R_Front_Knee']], p_curr[KP_MAP['R_Front_Hoof']]),
#         calculate_angle(p_curr[KP_MAP['L_Back_Stifle']], p_curr[KP_MAP['L_Back_Knee']], p_curr[KP_MAP['L_Back_Hoof']]),
#         calculate_angle(p_curr[KP_MAP['R_Back_Stifle']], p_curr[KP_MAP['R_Back_Knee']], p_curr[KP_MAP['R_Back_Hoof']]),
#     ])
#     features.extend(current_angles)
#     angles_buffer.append(current_angles)

#     if len(angles_buffer) < 2:
#         angular_velocities = np.zeros(4)
#     else:
#         angular_velocities = current_angles - angles_buffer[0]
#     features.extend(angular_velocities)

#     return np.array(features)


