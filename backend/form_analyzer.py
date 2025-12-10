from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

class FormAnalyzer:
    def __init__(self):
        """Initialize the YOLOv8 pose model"""
        print("Loading pose detection model...")
        self.model = YOLO('yolov8n-pose.pt')
        
    def analyze_squat(self, image_data):
        """
        Analyze squat form from an image
        
        Args:
            image_data: Image file (PIL Image, numpy array, or file path)
            
        Returns:
            dict: Analysis results with form feedback
        """
        # Run pose detection
        results = self.model(image_data)
        
        # Check if any people were detected
        if len(results[0].keypoints) == 0:
            return {
                'success': False,
                'error': 'No person detected in image',
                'feedback': []
            }
        
        # Get the first person's keypoints
        keypoints = results[0].keypoints.data[0].cpu().numpy()
        
        # Extract key body points (format: [x, y, confidence])
        left_shoulder = keypoints[5]   # Index 5
        right_shoulder = keypoints[6]  # Index 6
        left_hip = keypoints[11]       # Index 11
        right_hip = keypoints[12]      # Index 12
        left_knee = keypoints[13]      # Index 13
        right_knee = keypoints[14]     # Index 14
        left_ankle = keypoints[15]     # Index 15
        right_ankle = keypoints[16]    # Index 16
        
        # Use average of left and right sides
        shoulder = (left_shoulder[:2] + right_shoulder[:2]) / 2
        hip = (left_hip[:2] + right_hip[:2]) / 2
        knee = (left_knee[:2] + right_knee[:2]) / 2
        ankle = (left_ankle[:2] + right_ankle[:2]) / 2
        
        # Analyze form
        feedback = []
        
        # Check 1: Depth (hip should be below knee for proper squat)
        depth_good = hip[1] > knee[1]  # y-axis: higher value = lower in image
        if depth_good:
            feedback.append({
                'check': 'Depth',
                'status': 'PASS',
                'message': '✓ Good depth - hips below parallel'
            })
        else:
            feedback.append({
                'check': 'Depth',
                'status': 'FAIL',
                'message': '✗ Depth too shallow - squat deeper'
            })
        
        # Check 2: Back angle (torso shouldn't be too horizontal)
        back_angle = self._calculate_angle(shoulder, hip, knee)
        back_good = 45 < back_angle < 90
        if back_good:
            feedback.append({
                'check': 'Back Angle',
                'status': 'PASS',
                'message': f'✓ Good back angle ({back_angle:.0f}°)'
            })
        else:
            feedback.append({
                'check': 'Back Angle',
                'status': 'FAIL',
                'message': f'✗ Back too far forward ({back_angle:.0f}°) - stay upright'
            })
        
        # Check 3: Knee alignment (knees over ankles)
        knee_distance = abs(knee[0] - ankle[0])
        knee_good = knee_distance < 50  # pixels
        if knee_good:
            feedback.append({
                'check': 'Knee Alignment',
                'status': 'PASS',
                'message': '✓ Knees tracking well over feet'
            })
        else:
            feedback.append({
                'check': 'Knee Alignment',
                'status': 'WARNING',
                'message': '⚠ Watch knee position - keep over feet'
            })
        
        # Overall assessment
        all_pass = depth_good and back_good and knee_good
        
        return {
            'success': True,
            'all_pass': all_pass,
            'feedback': feedback,
            'keypoints': keypoints.tolist(),  # For visualization
            'num_people': len(results[0].keypoints)
        }
    
    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Calculate angle using dot product
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def annotate_image(self, image_path, analysis_result):
        """
        Draw annotations on image based on analysis
        
        Args:
            image_path: Path to original image
            analysis_result: Result from analyze_squat()
            
        Returns:
            Annotated image (PIL Image)
        """
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # If no person detected, return original
        if not analysis_result['success']:
            return Image.fromarray(img)
        
        # Draw keypoints and connections
        keypoints = np.array(analysis_result['keypoints'])
        
        # Draw skeleton connections
        connections = [
            (5, 11), (6, 12),  # Shoulder to hip
            (11, 13), (12, 14),  # Hip to knee
            (13, 15), (14, 16),  # Knee to ankle
            (5, 6), (11, 12)  # Left-right connections
        ]
        
        for connection in connections:
            pt1 = keypoints[connection[0]][:2].astype(int)
            pt2 = keypoints[connection[1]][:2].astype(int)
            
            # Color based on overall form
            color = (0, 255, 0) if analysis_result['all_pass'] else (255, 0, 0)
            cv2.line(img, tuple(pt1), tuple(pt2), color, 3)
        
        # Draw keypoints
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            color = (0, 255, 0) if analysis_result['all_pass'] else (255, 0, 0)
            cv2.circle(img, (x, y), 5, color, -1)
        
        # Add text feedback on image
        y_offset = 30
        for item in analysis_result['feedback']:
            status = item['status']
            message = item['message']
            
            # Color based on status
            if status == 'PASS':
                color = (0, 255, 0)  # Green
            elif status == 'FAIL':
                color = (255, 0, 0)  # Red
            else:
                color = (255, 255, 0)  # Yellow
            
            cv2.putText(img, message, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 35
        
        return Image.fromarray(img)

# Test if this file runs directly
if __name__ == "__main__":
    print("FormAnalyzer module loaded successfully!")
    analyzer = FormAnalyzer()
    print("Ready to analyze squat form!")

# **What this code does:**
# 1. Loads YOLOv8 pose model
# 2. Detects 17 keypoints on a person's body
# 3. Analyzes 3 aspects of squat form:
#    - **Depth**: Are hips below knees?
#    - **Back angle**: Is torso upright enough?
#    - **Knee alignment**: Are knees tracking over feet?
# 4. Returns pass/fail for each check
# 5. Can annotate images with visual feedback
