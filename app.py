import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import uuid
from pathlib import Path

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="RepCheck - AI Form Analysis",
    page_icon="💪",
    layout="wide"
)

# Custom CSS for mobile-friendly design
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 12px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .success-box {
        padding: 20px;
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fail-box {
        padding: 20px;
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for stats
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0

# FormAnalyzer class (backend logic merged in)
class FormAnalyzer:
    def __init__(self):
        """Initialize the YOLOv8 pose model"""
        self.model = None
        
    @st.cache_resource
    def load_model(_self):
        """Load model with caching"""
        return YOLO('yolov8n-pose.pt')
    
    def get_model(self):
        """Get cached model"""
        if self.model is None:
            self.model = self.load_model()
        return self.model
        
    def analyze_squat(self, image_data):
        """Analyze squat form from an image"""
        model = self.get_model()
        
        # Run pose detection
        results = model(image_data)
        
        # Check if any people were detected
        if len(results[0].keypoints) == 0:
            return {
                'success': False,
                'error': 'No person detected in image',
                'feedback': []
            }
        
        # Get the first person's keypoints
        keypoints = results[0].keypoints.data[0].cpu().numpy()
        
        # Extract key body points
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # Use average of left and right sides
        shoulder = (left_shoulder[:2] + right_shoulder[:2]) / 2
        hip = (left_hip[:2] + right_hip[:2]) / 2
        knee = (left_knee[:2] + right_knee[:2]) / 2
        ankle = (left_ankle[:2] + right_ankle[:2]) / 2
        
        # Analyze form
        feedback = []
        
        # Check 1: Depth
        depth_good = bool(hip[1] > knee[1])
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
        
        # Check 2: Back angle
        back_angle = self._calculate_angle(shoulder, hip, knee)
        back_good = bool(45 < back_angle < 90)
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
        
        # Check 3: Knee alignment
        knee_distance = float(abs(knee[0] - ankle[0]))
        knee_good = bool(knee_distance < 50)
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
        all_pass = bool(depth_good and back_good and knee_good)
        
        return {
            'success': True,
            'all_pass': all_pass,
            'feedback': feedback,
            'keypoints': keypoints.tolist(),
            'num_people': int(len(results[0].keypoints))
        }
    
    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return float(np.degrees(angle))
    
    def annotate_image(self, image, analysis_result):
        """Draw annotations on image"""
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if not analysis_result['success']:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        
        # Draw keypoints and connections
        keypoints = np.array(analysis_result['keypoints'])
        
        connections = [
            (5, 11), (6, 12),  # Shoulder to hip
            (11, 13), (12, 14),  # Hip to knee
            (13, 15), (14, 16),  # Knee to ankle
            (5, 6), (11, 12)  # Left-right connections
        ]
        
        for connection in connections:
            pt1 = keypoints[connection[0]][:2].astype(int)
            pt2 = keypoints[connection[1]][:2].astype(int)
            
            color = (0, 255, 0) if analysis_result['all_pass'] else (0, 0, 255)
            cv2.line(img, tuple(pt1), tuple(pt2), color, 3)
        
        # Draw keypoints
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            color = (0, 255, 0) if analysis_result['all_pass'] else (0, 0, 255)
            cv2.circle(img, (x, y), 5, color, -1)
        
        # Add text feedback
        y_offset = 30
        for item in analysis_result['feedback']:
            status = item['status']
            message = item['message']
            
            if status == 'PASS':
                color = (0, 255, 0)
            elif status == 'FAIL':
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)
            
            cv2.putText(img, message, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 35
        
        # Convert back to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return FormAnalyzer()

analyzer = get_analyzer()

# Title and description
st.title("💪 RepCheck")
st.markdown("### AI-Powered Lifting Form Analysis")
st.markdown("📸 Capture your squat • 📊 Get instant feedback • 📱 Works on any device")

st.divider()

# Create tabs
tab1, tab2, tab3 = st.tabs(["📸 Photo Analysis", "🎥 Video Upload", "📊 Stats"])

# Tab 1: Photo Analysis
with tab1:
    st.subheader("Analyze Your Squat Form")
    
    # Instructions
    with st.expander("📖 How to capture perfect squat photos", expanded=True):
        st.markdown("""
        **Setup:**
        1. 📱 Prop your phone/camera at hip height
        2. 📏 Position 3-4 feet away
        3. 📐 **Side angle is critical** (90° to your body)
        4. 💡 Ensure good lighting
        
        **Capture:**
        - Use a Bluetooth remote or self-timer
        - Get into squat position (bottom of squat)
        - Make sure your full body is visible
        """)
    
    st.divider()
    
    # Mode selection
    mode = st.radio("Choose how to add your photo:", 
                    ["📤 Upload from Gallery", "📸 Take Photo with Camera"],
                    horizontal=True)
    
    image_to_analyze = None
    
    if mode == "📤 Upload from Gallery":
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        image_to_analyze = uploaded_file
        
    else:
        st.info("💡 **Tip:** Use a Bluetooth remote or voice command to capture hands-free!")
        camera_photo = st.camera_input("📸 Position yourself and take photo")
        image_to_analyze = camera_photo
    
    # Display and analyze image
    if image_to_analyze:
        st.divider()
        st.subheader("Your Image")
        
        # Load image
        image = Image.open(image_to_analyze)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        
        with col2:
            st.markdown("*Annotated image will appear here after analysis*")
        
        # Analyze button
        if st.button("🔍 Analyze Form", key="analyze_btn", use_container_width=True, type="primary"):
            with st.spinner("Analyzing your form..."):
                try:
                    # Analyze
                    result = analyzer.analyze_squat(image)
                    
                    if result['success']:
                        # Increment counter
                        st.session_state.total_analyzed += 1
                        
                        # Show results
                        st.success("✅ Analysis Complete!")
                        
                        # Overall result
                        if result['all_pass']:
                            st.markdown('<div class="success-box"><h3>🎉 Excellent Form!</h3><p>All checks passed. Keep up the great work!</p></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="fail-box"><h3>⚠️ Form Issues Detected</h3><p>Review the feedback below to improve your form.</p></div>', unsafe_allow_html=True)
                        
                        # Detailed feedback
                        st.subheader("📋 Detailed Feedback")
                        
                        feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
                        
                        for idx, item in enumerate(result['feedback']):
                            status = item['status']
                            check = item['check']
                            message = item['message']
                            
                            with [feedback_col1, feedback_col2, feedback_col3][idx % 3]:
                                if status == 'PASS':
                                    st.success(f"**{check}**\n\n{message}")
                                elif status == 'FAIL':
                                    st.error(f"**{check}**\n\n{message}")
                                else:
                                    st.warning(f"**{check}**\n\n{message}")
                        
                        st.divider()
                        
                        # Show annotated image
                        st.subheader("📸 Annotated Image")
                        
                        col1_result, col2_result = st.columns([1, 1])
                        
                        with col1_result:
                            st.image(image, caption="Original", use_container_width=True)
                        
                        with col2_result:
                            annotated = analyzer.annotate_image(image, result)
                            st.image(annotated, caption="Form Analysis", use_container_width=True)
                            
                            # Download button
                            buf = io.BytesIO()
                            annotated.save(buf, format='PNG')
                            st.download_button(
                                label="💾 Download Annotated Image",
                                data=buf.getvalue(),
                                file_name="repcheck_analysis.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                    else:
                        st.error(f"❌ {result.get('error', 'Unknown error')}")
                        st.info("💡 Make sure a person is clearly visible from a **side angle**")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("🐛 Debug Info"):
                        st.code(traceback.format_exc())

# Tab 2: Video Upload
with tab2:
    st.subheader("🎥 Video Analysis")
    st.info("📹 Video analysis will allow frame-by-frame form checking. Coming soon!")
    
    video_file = st.file_uploader("Upload squat video (optional)", type=['mp4', 'mov'], key="video")
    
    if video_file:
        st.video(video_file)
        st.warning("⚠️ Video analysis feature is under development")

# Tab 3: Stats
with tab3:
    st.subheader("📊 Your Stats")
    
    st.metric("Total Images Analyzed", st.session_state.total_analyzed, 
              help="Number of images you've analyzed in this session")
    
    st.progress(min(st.session_state.total_analyzed / 50, 1.0))
    st.caption(f"Goal: Analyze 50 reps")
    
    if st.session_state.total_analyzed > 0:
        st.balloons()

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>RepCheck v1.0</strong> | AI-Powered Form Analysis</p>
        <p>💪 Lift safer, lift better</p>
        <p style='font-size: 12px; margin-top: 10px;'>
            Built with YOLOv8 and Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)
