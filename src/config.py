import torch

# --- Model Paths ---
YOLO_MODEL_PATH = "/fs/nexus-scratch/hwahed/yoloFineTuning/Horse_Keypoints/goodRun!/weights/best.pt"
TRANSFORMER_MODEL_PATH = '/fs/nexus-scratch/hwahed/GaitAnalyzer/top_models_onecycle/gait_model_rank_1_epoch_26_acc_92.86.pth'

# --- Analysis & Video Settings ---
FRAME_SEQUENCE_LENGTH = 20
CLASSES = ['standing', 'walking', 'trotting', 'cantering', 'gallop']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_VIDEO_DIR = '/fs/nexus-scratch/hwahed/ai_equestrian/CoachVideos/'

# --- Rider Grading Criteria ---
# NEW: Bounding Box Aspect Ratio Method
# We determine the horse's orientation by the aspect ratio of its body's bounding box.
# A low height/width ratio indicates a side-on view suitable for grading.
MAX_ASPECT_RATIO = 0.5 # Lower is stricter. A horse is wider than it is tall.
IDEAL_ELBOW_ANGLE = 140 # The target angle for the rider's elbow
# Constant for the original rider guidance logic
ANGLE_CONSTANT = 83
DEBUG = 0

# --- Transformer Model Hyperparameters ---
RAW_INPUT_SIZE = 79
N_HEADS = 8
D_MODEL = N_HEADS * 11  # 88
N_LAYERS = 7
DROPOUT_PROB = 0.2
NUM_CLASSES = len(CLASSES)

# --- Keypoint Definitions ---
# HORSE keypoints start at index 15
HORSE_LEG_KEYPOINT_INDICES = list(range(15, 27))

# Keypoints for the body aspect ratio calculation (torso and legs, no head/neck)
HORSE_BODY_AND_LEGS_INDICES = list(range(23, 27)) + list(range(28, 30)) + list(range(32, 37))

# Mapping of leg keypoints to their index within the sliced array (0-11)
KP_MAP = {
    'L_Front_Hoof': 0, 'R_Front_Hoof': 1, 'L_Back_Hoof': 2, 'R_Back_Hoof': 3,
    'L_Front_Knee': 4, 'R_Front_Knee': 5, 'L_Back_Knee': 6, 'R_Back_Knee': 7,
    'L_Front_Shoulder': 8, 'R_Front_Shoulder': 9,
    'R_Back_Stifle': 10, 'L_Back_Stifle': 11
}

# Indices for specific keypoints from the full YOLO output
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
RIGHT_BACK_STIFLE_IDX = 25
LEFT_BACK_STIFLE_IDX = 26
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

#some basic CV2 colours
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
LIGHT_GREEN = (144, 238, 144)
ORANGE = (227, 137, 20)
LIGHT_ORANGE = (222, 171, 104)