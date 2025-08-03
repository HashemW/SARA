import cv2
import torch
import numpy as np
import torch.nn as nn
from ultralytics import YOLO
from collections import deque
import math

YOLO_MODEL_PATH = "/fs/nexus-scratch/hwahed/yoloFineTuning/Horse_Keypoints/goodRun!/weights/best.pt" 
GAIT_MODEL_PATH = "/fs/nexus-scratch/hwahed/GaitAnalyzer/model_results/96.97accuracy.pth"
INPUT_VIDEO_PATH = "/fs/nexus-scratch/hwahed/GaitAnalyzer/testVideos/arabStances.mp4"
OUTPUT_VIDEO_PATH = "output_video.mp4"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class GaitTransformer(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        
        # This linear layer projects your 87 features into the model's dimension (d_model)
        self.input_projection = nn.Linear(num_features, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.d_model = d_model
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, num_features]
        """
        # Project input features to model dimension
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through the transformer encoder
        output = self.transformer_encoder(src)
        
        # We use the output of the first token ([CLS] token style) for classification
        output = output[:, 0, :]
        
        # Final classification
        output = self.classifier(output)
        return output
    
SEQUENCE_LENGTH = 20
NUM_FEATURES = 87
NUM_CLASSES = 5
HORSE_LEG_KEYPOINT_INDICES = list(range(15, 27))
LABEL_TO_GAIT = {0: 'Walking', 1: 'Trotting', 2: 'Cantering', 3: 'Gallop', 4: 'Standing'}
KP_MAP = {
    'L_Front_Hoof': 0, 'R_Front_Hoof': 1, 'L_Back_Hoof': 2, 'R_Back_Hoof': 3,
    'L_Front_Knee': 4, 'R_Front_Knee': 5, 'L_Back_Knee': 6, 'R_Back_Knee': 7,
    'L_Front_Shoulder': 8, 'R_Front_Shoulder': 9,
    'R_Back_Stifle': 10, 'L_Back_Stifle': 11
}

# --- Copy your feature calculation functions here ---
# It's crucial that these are IDENTICAL to the ones used for training
def calculate_angle(p1, p2, p3):
    v1 = p1 - p2; v2 = p3 - p2
    dot_product = np.dot(v1, v2); norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0: return 180.0
    angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    return np.degrees(angle)

# ... include all other feature functions like calculate_orientation_features, euclidean_dist etc. ...
# For now, let's assume a placeholder for the full feature calculation
def euclidean_dist(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# Ensure calculate_angle is also present, which it already is.

def calculate_all_features(p_curr, p_prev, p_prev_prev, angles_buffer):
    """
    Calculates the 87-feature vector, matching the training script EXACTLY.
    """
    features = []

    # 1. Base Features (Coordinates, Velocity, Acceleration)
    x_coords, y_coords = p_curr[:, 0], p_curr[:, 1]
    features.extend([np.max(x_coords) - np.min(x_coords), np.max(y_coords) - np.min(y_coords)])
    features.extend(p_curr.flatten())
    velocities = p_curr - p_prev
    features.extend(velocities.flatten())
    prev_velocities = p_prev - p_prev_prev
    features.extend((velocities - prev_velocities).flatten())

    # 2. Distance & Oscillation Features
    features.append(euclidean_dist(p_curr[KP_MAP['L_Front_Hoof']], p_curr[KP_MAP['R_Front_Hoof']]))
    features.append(euclidean_dist(p_curr[KP_MAP['L_Back_Hoof']], p_curr[KP_MAP['R_Back_Hoof']]))
    features.append(euclidean_dist(p_curr[KP_MAP['L_Front_Hoof']], p_curr[KP_MAP['L_Back_Hoof']]))
    features.append(euclidean_dist(p_curr[KP_MAP['R_Front_Hoof']], p_curr[KP_MAP['R_Back_Hoof']]))
    features.append(np.mean(p_curr[:, 1]) - np.mean(p_prev[:, 1]))

    # 3. Orientation Features (COMMENTED OUT to match the 87-feature model)
    # orientation_features = calculate_orientation_features(p_curr)
    # features.extend(orientation_features)

    # 4. Angle and Angular Velocity Features
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

    return np.array(features)


# --- Main Visualization Logic ---
