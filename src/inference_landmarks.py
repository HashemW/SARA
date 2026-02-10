import torch
import torch.nn as nn
import numpy as np
import config
# --- MODEL DEFINITION (Must match training) ---
class LandmarkNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8) # 4 points * 2 coords
        )
    def forward(self, x): return self.model(x)

# --- INFERENCE CLASS ---
class VirtualLandmarkPredictor:
    def __init__(self, model_path=config.LANDMARK_MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # INDICES (Must match the CORE_BODYPARTS used in training)
        # 1. neck_base (32?), 2. throat (27?), 3. back_base (33?), 4. back_end (36?)
        # 5-8. Thighs (23,24,25,26)
        # 9-12. Knees (19,20,21,22)
        # CHECK YOUR YOLO MAPPING! This list must be exact.
        self.indices = [
            32, 27, 33, 36, 
            23, 24, 25, 26,
            19, 20, 21, 22
        ]
        
        # Input dim = Bbox(3) + Keypoints(12*3) = 39
        self.model = LandmarkNet(39).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, keypoints, bbox):
        """
        Input: keypoints (N, 3), bbox [xc, yc, w, h]
        Output: Dict of 4 points { 'back':(x,y), 'belly':(x,y), 'left':(x,y), 'right':(x,y) }
        """
        bx, by, bw, bh = bbox
        x_min, y_min = bx - bw/2, by - bh/2
        
        # 1. Normalize Inputs
        feats = [bw, bh, bw/bh if bh>0 else 0]
        for idx in self.indices:
            if idx < len(keypoints):
                kp = keypoints[idx]
                nx = (kp[0] - x_min) / bw if bw>0 else 0
                ny = (kp[1] - y_min) / bh if bh>0 else 0
                feats.extend([nx, ny, kp[2]])
            else:
                feats.extend([0,0,0])
                
        input_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 2. Run Model
        with torch.no_grad():
            # Output is [back_x, back_y, belly_x, belly_y, left_x, left_y, right_x, right_y]
            out = self.model(input_tensor).cpu().numpy()[0]
            
        # 3. Denormalize Outputs (0-1 -> Pixels)
        def denorm(x, y):
            return (int(x * bw + x_min), int(y * bh + y_min))

        return {
            'back':  denorm(out[0], out[1]),
            'belly': denorm(out[2], out[3]),
            'left':  denorm(out[4], out[5]),
            'right': denorm(out[6], out[7])
        }