import torch
import os
from enum import Enum

# --- PATH SETUP (Dynamic) ---
# Get the folder where THIS config.py file is located (e.g., .../src)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if os.path.exists("/data"):
    OUTPUT_VIDEO_DIR = '/data/results'
else:
    PARENTPARENT_DIR = os.path.dirname(PARENT_DIR)
    OUTPUT_VIDEO_DIR = os.path.join(PARENTPARENT_DIR, "outputs")

MODEL_ROOT = PARENT_DIR

os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
# 1. Model Paths
# This assumes 'best.pt' and the .pth file are inside the 'src' folder
YOLO_MODEL_PATH = os.path.join(MODEL_ROOT, "best.pt")
GAIT_MODEL_PATH = os.path.join(MODEL_ROOT, "best_gait_transformer_lean.pth")
ORIENTATION_MODEL_PATH = os.path.join(MODEL_ROOT, "horse_orientation_best.pth")
LANDMARK_MODEL_PATH = os.path.join(MODEL_ROOT, "horse_virtual_landmarks_best.pth")
VERTICAL_MODEL_PATH = os.path.join(MODEL_ROOT, "horse_vertical_vector_best.pth")


# --- Analysis & Video Settings ---
FRAME_SEQUENCE_LENGTH = 30
CLASSES = ['standing', 'walking', 'trotting', 'cantering', 'gallop']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Rider Grading Criteria ---
# --- Rider Grading Criteria ---
MAX_ASPECT_RATIO = 0.5 
IDEAL_ELBOW_ANGLE = 140
ANGLE_CONSTANT = 83
DEBUG = 0

# --- Segmentation Settings ---
YOLO_PERSON_CONF_THRESHOLD = 0.60
SAM_CONF_THRESHOLD = 0.80

# --- Transformer Model Hyperparameters (UPDATED) ---
RAW_INPUT_SIZE = 22     # UPDATED: 98 Features
N_HEADS = 2
D_MODEL = 64           # UPDATED: Matches training script
N_LAYERS = 3           # UPDATED: Matches training script
DROPOUT_PROB = 0.3      # UPDATED: Matches training script
NUM_CLASSES = len(CLASSES)

# --- Keypoint Definitions ---
RIDER_KEYPOINT_INDICES = list(range(1, 15))
HORSE_LEG_KEYPOINT_INDICES = list(range(15, 27))
HORSE_BODY_AND_LEGS_INDICES = list(range(23, 27)) + list(range(28, 30)) + list(range(32, 37))

KP_MAP = {
    'L_Front_Hoof': 0, 'R_Front_Hoof': 1, 'L_Back_Hoof': 2, 'R_Back_Hoof': 3,
    'L_Front_Knee': 4, 'R_Front_Knee': 5, 'L_Back_Knee': 6, 'R_Back_Knee': 7,
    'L_Front_Shoulder': 8, 'R_Front_Shoulder': 9,
    'R_Back_Stifle': 10, 'L_Back_Stifle': 11
}

# Keypoint Indices
NOSE_IDX = 0
RIDER_L_EAR_IDX = 1
RIDER_R_EAR_IDX = 2
RIDER_L_SHOULDER_IDX = 3
RIDER_R_SHOULDER_IDX = 4
RIDER_L_ELBOW_IDX = 5
RIDER_R_ELBOW_IDX = 6
RIDER_L_WRIST_IDX = 7
RIDER_R_WRIST_IDX = 8
RIDER_L_HIP_IDX = 9
RIDER_R_HIP_IDX = 10
RIDER_L_KNEE_IDX = 11
RIDER_R_KNEE_IDX = 12
RIDER_L_ANKLE_IDX = 13
RIDER_R_ANKLE_IDX = 14

# Horse keypoints indices
LEFT_FRONT_HOOF_IDX = 15
RIGHT_FRONT_HOOF_IDX = 16
LEFT_BACK_HOOF_IDX = 17
RIGHT_BACK_HOOF_IDX = 18
LEFT_FRONT_KNEE_IDX = 19
RIGHT_FRONT_KNEE_IDX = 20
LEFT_BACK_KNEE_IDX = 21
RIGHT_BACK_KNEE_IDX = 22
LEFT_FRONT_SHOULDER_IDX = 23
RIGHT_FRONT_SHOULDER_IDX = 24
LEFT_BACK_STIFLE_IDX = 25
RIGHT_BACK_STIFLE_IDX = 26
MOUTH_IDX = 27
LEFT_EYE_IDX = 28
RIGHT_EYE_IDX = 29
LEFT_EAR_IDX = 30
RIGHT_EAR_IDX = 31
MIDPOINT_WITHERS_CHEST_IDX = 32
LEFT_WITHERS_IDX = 33
CHEST_IDX = 34
RIGHT_WITHERS_IDX = 35
TOP_OF_TAIL_IDX = 36

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
LIGHT_GREEN = (144, 238, 144)
ORANGE = (227, 137, 20)
LIGHT_ORANGE = (222, 171, 104)

class Direction(Enum):
    RIGHT_TO_LEFT = 1
    LEFT_TO_RIGHT = 2
    BOTTOM_RIGHT_TO_TOP_LEFT = 3
    TOP_RIGHT_TO_BOTTOM_LEFT = 4
    BOTTOM_LEFT_TO_TOP_RIGHT = 5
    TOP_LEFT_TO_BOTTOM_RIGHT = 6
    BACK_VIEW = 7
    FRONT_VIEW = 8