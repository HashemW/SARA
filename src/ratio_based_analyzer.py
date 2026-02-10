import numpy as np
import cv2
from enum import Enum

class ViewType(Enum):
    """View categories based on hip-to-back ratio"""
    SIDE = "side"              # ratio <= 0.15 - your current working case
    QUARTER = "three_quarter"  # 0.15 < ratio <= 0.30
    FRONT_BACK = "front_back"  # ratio > 0.30
    UNKNOWN = "unknown"


class RatioBasedPostureAnalyzer:
    """
    Posture analysis using hip-to-back ratio to determine view and appropriate checks.
    
    The key insight: The ratio tells us HOW MUCH of the rider/horse we're seeing:
    - Low ratio = seeing the SIDE (thin hips, long back)
    - Medium ratio = seeing at an ANGLE (both partially foreshortened)  
    - High ratio = seeing FRONT/BACK (wide hips, compressed back)
    """
    
    # Thresholds tuned from your data - adjust based on your videos
    SIDE_VIEW_THRESHOLD = 0.15      # Your current threshold
    QUARTER_VIEW_THRESHOLD = 0.30   # Transition to front/back views
    
    def __init__(self):
        # Store ratio history for smoothing (prevents jittery classifications)
        self.ratio_history = []
        self.max_history = 10
    
    def calculate_hip_to_back_ratio(self, keypoints_xy, confidences):
        """
        Your existing ratio calculation - extracted for clarity.
        
        Returns:
            tuple: (ratio, rider_hip_distance, horse_back_distance, all_visible)
        """
        from config import (RIDER_L_HIP_IDX, RIDER_R_HIP_IDX,
                           LEFT_WITHERS_IDX, RIGHT_WITHERS_IDX, TOP_OF_TAIL_IDX)
        
        # Check keypoint confidence
        kp_indices = [RIDER_L_HIP_IDX, RIDER_R_HIP_IDX, 
                      LEFT_WITHERS_IDX, RIGHT_WITHERS_IDX, TOP_OF_TAIL_IDX]
        confs = confidences[kp_indices]
        
        CONF_THRESHOLD = 0.6
        if not np.all(confs > CONF_THRESHOLD):
            return None, None, None, False
        
        # Get keypoint positions
        l_hip_pt = keypoints_xy[RIDER_L_HIP_IDX]
        r_hip_pt = keypoints_xy[RIDER_R_HIP_IDX]
        l_wither_pt = keypoints_xy[LEFT_WITHERS_IDX]
        r_wither_pt = keypoints_xy[RIGHT_WITHERS_IDX]
        tail_top_pt = keypoints_xy[TOP_OF_TAIL_IDX]
        
        # Calculate distances
        rider_hip_distance = np.linalg.norm(l_hip_pt - r_hip_pt)
        wither_midpoint = (l_wither_pt + r_wither_pt) / 2.0
        horse_back_distance = np.linalg.norm(wither_midpoint - tail_top_pt)
        
        if horse_back_distance < 1e-6:  # Avoid division by zero
            return None, rider_hip_distance, horse_back_distance, False
        
        ratio = rider_hip_distance / horse_back_distance
        
        return ratio, rider_hip_distance, horse_back_distance, True
    
    def classify_view_from_ratio(self, ratio):
        """
        Determine view type from the ratio value.
        
        This is where we interpret what the ratio is telling us about perspective.
        """
        if ratio is None:
            return ViewType.UNKNOWN
        
        # Add to history for temporal smoothing
        self.ratio_history.append(ratio)
        if len(self.ratio_history) > self.max_history:
            self.ratio_history.pop(0)
        
        # Use smoothed ratio to avoid jitter
        smoothed_ratio = np.median(self.ratio_history)
        
        # Classify based on thresholds
        if smoothed_ratio <= self.SIDE_VIEW_THRESHOLD:
            return ViewType.SIDE
        elif smoothed_ratio <= self.QUARTER_VIEW_THRESHOLD:
            return ViewType.QUARTER
        else:
            return ViewType.FRONT_BACK
    
    def analyze_frame(self, keypoints_xy, confidences):
        """
        Main analysis function - calculates ratio and applies appropriate checks.
        
        Returns:
            dict with analysis results and guidance for drawing
        """
        # Calculate the ratio
        ratio, hip_dist, back_dist, all_visible = \
            self.calculate_hip_to_back_ratio(keypoints_xy, confidences)
        
        if not all_visible:
            return {
                'view': ViewType.UNKNOWN,
                'ratio': None,
                'should_draw_guidance': False,
                'feedback': "Insufficient keypoints detected"
            }
        
        # Classify view
        view_type = self.classify_view_from_ratio(ratio)
        
        # Apply view-specific analysis
        if view_type == ViewType.SIDE:
            result = self._analyze_side_view(keypoints_xy, confidences, ratio)
        elif view_type == ViewType.QUARTER:
            result = self._analyze_quarter_view(keypoints_xy, confidences, ratio)
        elif view_type == ViewType.FRONT_BACK:
            result = self._analyze_front_back_view(keypoints_xy, confidences, ratio)
        else:
            result = {
                'should_draw_guidance': False,
                'feedback': "Cannot analyze posture"
            }
        
        # Add common info to result
        result['view'] = view_type
        result['ratio'] = ratio
        result['hip_distance'] = hip_dist
        result['back_distance'] = back_dist
        
        return result
    
    def _analyze_side_view(self, keypoints_xy, confidences, ratio):
        """
        Side view analysis - THIS IS YOUR EXISTING WORKING CODE!
        
        In side view:
        - Ratio is small (thin hips, long back)
        - We can see full body profile
        - Check: vertical alignment (ear-hip-ankle), perpendicular to horse's back
        """
        return {
            'should_draw_guidance': True,
            'guidance_type': 'full_geometric',  # Use your existing draw_rider_guidance
            'feedback': f"Side view (ratio: {ratio:.3f}) - Checking vertical alignment",
            'checks': ['back_alignment', 'leg_position', 'arm_angles']
        }
    
    def _analyze_quarter_view(self, keypoints_xy, confidences, ratio):
        """
        3/4 view analysis - partial visibility, limited checks.
        
        In quarter view:
        - Ratio is medium (both measurements partially foreshortened)
        - One side is more visible than the other
        - Check: what we CAN see reliably
        
        What's reliable in 3/4 view?
        1. Hip levelness (should be relatively equal height)
        2. Head position (should be over the body center)
        3. Upper body symmetry
        """
        from config import (RIDER_L_HIP_IDX, RIDER_R_HIP_IDX,
                           RIDER_L_SHOULDER_IDX, RIDER_R_SHOULDER_IDX,
                           NOSE_IDX)
        
        feedback_items = []
        checks = {}
        
        # Check 1: Hip levelness
        if confidences[RIDER_L_HIP_IDX] > 0.5 and confidences[RIDER_R_HIP_IDX] > 0.5:
            left_hip = keypoints_xy[RIDER_L_HIP_IDX]
            right_hip = keypoints_xy[RIDER_R_HIP_IDX]
            hip_diff = abs(left_hip[1] - right_hip[1])
            
            # Hip difference should be small (< 20 pixels typically)
            hip_level_good = hip_diff < 25
            checks['hip_level'] = hip_level_good
            
            if not hip_level_good:
                feedback_items.append("Level your hips")
        
        # Check 2: Shoulder levelness  
        if confidences[RIDER_L_SHOULDER_IDX] > 0.5 and confidences[RIDER_R_SHOULDER_IDX] > 0.5:
            left_shoulder = keypoints_xy[RIDER_L_SHOULDER_IDX]
            right_shoulder = keypoints_xy[RIDER_R_SHOULDER_IDX]
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
            
            shoulder_level_good = shoulder_diff < 25
            checks['shoulder_level'] = shoulder_level_good
            
            if not shoulder_level_good:
                feedback_items.append("Level your shoulders")
        
        feedback = f"3/4 view (ratio: {ratio:.3f})"
        if feedback_items:
            feedback += " - " + ", ".join(feedback_items)
        else:
            feedback += " - Good balance!"
        
        return {
            'should_draw_guidance': len(checks) > 0,
            'guidance_type': 'symmetry',  # Different drawing style
            'feedback': feedback,
            'checks': checks
        }
    
    def _analyze_front_back_view(self, keypoints_xy, confidences, ratio):
        """
        Front/back view analysis - focus entirely on symmetry.
        
        In front/back view:
        - Ratio is large (wide hips, compressed back)
        - Both sides equally visible
        - Check: perfect symmetry is key
        
        What matters in front/back view?
        1. Shoulders level (no tilting)
        2. Hips level (no collapsing to one side)
        3. Hands at equal height
        4. Centered over horse (head between shoulders, not leaning)
        """
        from config import (RIDER_L_HIP_IDX, RIDER_R_HIP_IDX,
                           RIDER_L_SHOULDER_IDX, RIDER_R_SHOULDER_IDX,
                           RIDER_L_WRIST_IDX, RIDER_R_WRIST_IDX,
                           NOSE_IDX)
        
        feedback_items = []
        checks = {}
        
        # Check 1: Shoulder symmetry (CRITICAL in front/back view)
        if confidences[RIDER_L_SHOULDER_IDX] > 0.5 and confidences[RIDER_R_SHOULDER_IDX] > 0.5:
            left_shoulder = keypoints_xy[RIDER_L_SHOULDER_IDX]
            right_shoulder = keypoints_xy[RIDER_R_SHOULDER_IDX]
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
            
            # In front view, shoulders should be VERY level
            shoulder_good = shoulder_diff < 15
            checks['shoulder_symmetry'] = shoulder_good
            
            if not shoulder_good:
                feedback_items.append("Level shoulders")
        
        # Check 2: Hip symmetry (CRITICAL for balance)
        if confidences[RIDER_L_HIP_IDX] > 0.5 and confidences[RIDER_R_HIP_IDX] > 0.5:
            left_hip = keypoints_xy[RIDER_L_HIP_IDX]
            right_hip = keypoints_xy[RIDER_R_HIP_IDX]
            hip_diff = abs(left_hip[1] - right_hip[1])
            
            hip_good = hip_diff < 15
            checks['hip_symmetry'] = hip_good
            
            if not hip_good:
                feedback_items.append("Level hips")
        
        # Check 3: Hand height equality
        if confidences[RIDER_L_WRIST_IDX] > 0.5 and confidences[RIDER_R_WRIST_IDX] > 0.5:
            left_wrist = keypoints_xy[RIDER_L_WRIST_IDX]
            right_wrist = keypoints_xy[RIDER_R_WRIST_IDX]
            wrist_diff = abs(left_wrist[1] - right_wrist[1])
            
            hands_good = wrist_diff < 30
            checks['hand_symmetry'] = hands_good
            
            if not hands_good:
                feedback_items.append("Even hand height")
        
        # Check 4: Centeredness (is the rider leaning?)
        if (confidences[NOSE_IDX] > 0.5 and 
            confidences[RIDER_L_SHOULDER_IDX] > 0.5 and 
            confidences[RIDER_R_SHOULDER_IDX] > 0.5):
            
            nose = keypoints_xy[NOSE_IDX]
            left_shoulder = keypoints_xy[RIDER_L_SHOULDER_IDX]
            right_shoulder = keypoints_xy[RIDER_R_SHOULDER_IDX]
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            
            # Head should be centered between shoulders
            lean_amount = abs(nose[0] - shoulder_center_x)
            centered = lean_amount < 30
            checks['centered'] = centered
            
            if not centered:
                feedback_items.append("Center over horse")
        
        feedback = f"Front/back view (ratio: {ratio:.3f})"
        if feedback_items:
            feedback += " - " + ", ".join(feedback_items)
        else:
            feedback += " - Excellent symmetry!"
        
        return {
            'should_draw_guidance': len(checks) > 0,
            'guidance_type': 'symmetry',
            'feedback': feedback,
            'checks': checks
        }


def draw_ratio_based_guidance(frame, analysis_result, keypoints_xy, confidences):
    """
    Draws appropriate visual guidance based on the analysis result.
    
    This is where we visualize differently for each view type.
    """
    if not analysis_result['should_draw_guidance']:
        return
    
    view = analysis_result['view']
    ratio = analysis_result['ratio']
    
    # Always show the ratio and view type
    color = (0, 255, 0) if ratio <= 0.15 else (0, 165, 255) if ratio <= 0.30 else (0, 0, 255)
    cv2.putText(frame, f"Ratio: {ratio:.3f} - {view.value}", 
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, analysis_result['feedback'], 
                (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    guidance_type = analysis_result.get('guidance_type')
    
    if guidance_type == 'full_geometric':
        # SIDE VIEW - use your existing sophisticated geometric drawing
        # This is where draw_rider_guidance() gets called
        cv2.putText(frame, "Using full geometric guidance", 
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
    elif guidance_type == 'symmetry':
        # QUARTER or FRONT/BACK VIEW - draw symmetry indicators
        draw_symmetry_indicators(frame, analysis_result, keypoints_xy, confidences)


def draw_symmetry_indicators(frame, analysis_result, keypoints_xy, confidences):
    """
    Draws symmetry checks for quarter and front/back views.
    
    Visual indicators:
    - Green lines between symmetric points if good
    - Red lines if asymmetric
    - Horizontal reference lines to show levelness
    """
    from config import (RIDER_L_HIP_IDX, RIDER_R_HIP_IDX,
                       RIDER_L_SHOULDER_IDX, RIDER_R_SHOULDER_IDX,
                       RIDER_L_WRIST_IDX, RIDER_R_WRIST_IDX,
                       GREEN, RED, YELLOW)
    
    checks = analysis_result.get('checks', {})
    
    # Draw hip level indicator
    if 'hip_level' in checks or 'hip_symmetry' in checks:
        if confidences[RIDER_L_HIP_IDX] > 0.5 and confidences[RIDER_R_HIP_IDX] > 0.5:
            left_hip = keypoints_xy[RIDER_L_HIP_IDX].astype(int)
            right_hip = keypoints_xy[RIDER_R_HIP_IDX].astype(int)
            
            hip_good = checks.get('hip_level', checks.get('hip_symmetry', False))
            color = GREEN if hip_good else RED
            
            # Draw line between hips
            cv2.line(frame, tuple(left_hip), tuple(right_hip), color, 3)
            
            # Draw horizontal reference through midpoint
            midpoint_x = (left_hip[0] + right_hip[0]) // 2
            midpoint_y = (left_hip[1] + right_hip[1]) // 2
            cv2.line(frame, (midpoint_x - 50, midpoint_y), 
                    (midpoint_x + 50, midpoint_y), YELLOW, 1, cv2.LINE_AA)
    
    # Draw shoulder level indicator
    if 'shoulder_level' in checks or 'shoulder_symmetry' in checks:
        if confidences[RIDER_L_SHOULDER_IDX] > 0.5 and confidences[RIDER_R_SHOULDER_IDX] > 0.5:
            left_shoulder = keypoints_xy[RIDER_L_SHOULDER_IDX].astype(int)
            right_shoulder = keypoints_xy[RIDER_R_SHOULDER_IDX].astype(int)
            
            shoulder_good = checks.get('shoulder_level', checks.get('shoulder_symmetry', False))
            color = GREEN if shoulder_good else RED
            
            cv2.line(frame, tuple(left_shoulder), tuple(right_shoulder), color, 3)
            
            midpoint_x = (left_shoulder[0] + right_shoulder[0]) // 2
            midpoint_y = (left_shoulder[1] + right_shoulder[1]) // 2
            cv2.line(frame, (midpoint_x - 50, midpoint_y), 
                    (midpoint_x + 50, midpoint_y), YELLOW, 1, cv2.LINE_AA)
    
    # Draw hand symmetry indicator
    if 'hand_symmetry' in checks:
        if confidences[RIDER_L_WRIST_IDX] > 0.5 and confidences[RIDER_R_WRIST_IDX] > 0.5:
            left_wrist = keypoints_xy[RIDER_L_WRIST_IDX].astype(int)
            right_wrist = keypoints_xy[RIDER_R_WRIST_IDX].astype(int)
            
            hands_good = checks['hand_symmetry']
            color = GREEN if hands_good else RED
            
            # Draw circles on wrists and connecting line
            cv2.circle(frame, tuple(left_wrist), 6, color, -1)
            cv2.circle(frame, tuple(right_wrist), 6, color, -1)
            cv2.line(frame, tuple(left_wrist), tuple(right_wrist), color, 2, cv2.LINE_AA)