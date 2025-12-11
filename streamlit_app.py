import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Page config
st.set_page_config(
    page_title="RepCheck - AI Form Analysis",
    page_icon="💪",
    layout="wide"
)

# Custom CSS
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

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# FormAnalyzer class
class FormAnalyzer:
    def __init__(self):
        self.model = None
        
    @st.cache_resource
    def load_model(_self):
        return YOLO('yolov8n-pose.pt')
    
    def get_model(self):
        if self.model is None:
            self.model = self.load_model()
        return self.model
        
    def analyze_squat(self, image_data):
        model = self.get_model()
        results = model(image_data)
        
        if len(results[0].keypoints) == 0:
            return {
                'success': False,
                'error': 'No person detected in image',
                'feedback': []
            }
        
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
        
        shoulder = (left_shoulder[:2] + right_shoulder[:2]) / 2
        hip = (left_hip[:2] + right_hip[:2]) / 2
        knee = (left_knee[:2] + right_knee[:2]) / 2
        ankle = (left_ankle[:2] + right_ankle[:2]) / 2
        
        feedback = []
        checks = {}
        
        # Check 1: Depth
        depth_good = bool(hip[1] > knee[1])
        checks['depth'] = depth_good
        feedback.append({
            'check': 'Depth',
            'status': 'PASS' if depth_good else 'FAIL',
            'message': '✓ Good depth - hips below parallel' if depth_good else '✗ Depth too shallow - squat deeper'
        })
        
        # Check 2: Back angle
        back_angle = self._calculate_angle(shoulder, hip, knee)
        back_good = bool(45 < back_angle < 90)
        checks['back_angle'] = back_good
        checks['back_angle_value'] = float(back_angle)
        feedback.append({
            'check': 'Back Angle',
            'status': 'PASS' if back_good else 'FAIL',
            'message': f'✓ Good back angle ({back_angle:.0f}°)' if back_good else f'✗ Back too far forward ({back_angle:.0f}°) - stay upright'
        })
        
        # Check 3: Knee alignment
        knee_distance = float(abs(knee[0] - ankle[0]))
        knee_good = bool(knee_distance < 50)
        checks['knee_alignment'] = knee_good
        feedback.append({
            'check': 'Knee Alignment',
            'status': 'PASS' if knee_good else 'WARNING',
            'message': '✓ Knees tracking well over feet' if knee_good else '⚠ Watch knee position - keep over feet'
        })
        
        all_pass = bool(depth_good and back_good and knee_good)
        
        return {
            'success': True,
            'all_pass': all_pass,
            'feedback': feedback,
            'checks': checks,
            'keypoints': keypoints.tolist(),
            'num_people': int(len(results[0].keypoints)),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_angle(self, point1, point2, point3):
        vector1 = point1 - point2
        vector2 = point3 - point2
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return float(np.degrees(angle))
    
    def annotate_image(self, image, analysis_result):
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if not analysis_result['success']:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        
        keypoints = np.array(analysis_result['keypoints'])
        
        connections = [
            (5, 11), (6, 12),
            (11, 13), (12, 14),
            (13, 15), (14, 16),
            (5, 6), (11, 12)
        ]
        
        for connection in connections:
            pt1 = keypoints[connection[0]][:2].astype(int)
            pt2 = keypoints[connection[1]][:2].astype(int)
            color = (0, 255, 0) if analysis_result['all_pass'] else (0, 0, 255)
            cv2.line(img, tuple(pt1), tuple(pt2), color, 3)
        
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            color = (0, 255, 0) if analysis_result['all_pass'] else (0, 0, 255)
            cv2.circle(img, (x, y), 5, color, -1)
        
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
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

# PDF Generation
def generate_pdf_report(history):
    """Generate comprehensive PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E7D32'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("💪 RepCheck Progress Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Summary stats
    total_reps = len(history)
    passed_reps = sum(1 for h in history if h['all_pass'])
    pass_rate = (passed_reps / total_reps * 100) if total_reps > 0 else 0
    
    # Count issues
    depth_fails = sum(1 for h in history if not h['checks'].get('depth', True))
    back_fails = sum(1 for h in history if not h['checks'].get('back_angle', True))
    knee_warns = sum(1 for h in history if not h['checks'].get('knee_alignment', True))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Reps Analyzed', str(total_reps)],
        ['Reps with Perfect Form', str(passed_reps)],
        ['Overall Pass Rate', f'{pass_rate:.1f}%'],
        ['', ''],
        ['Form Issues Detected', ''],
        ['Depth Failures', str(depth_fails)],
        ['Back Angle Failures', str(back_fails)],
        ['Knee Alignment Warnings', str(knee_warns)],
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 5), (-1, 5), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 5), (-1, 5), colors.lightgrey),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.5*inch))
    
    # Recommendations
    story.append(Paragraph("📋 Personalized Recommendations", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    recommendations = []
    if depth_fails > total_reps * 0.3:
        recommendations.append("• Focus on hip mobility - you're not reaching parallel depth consistently")
    if back_fails > total_reps * 0.3:
        recommendations.append("• Work on core strength and keeping chest up during descent")
    if knee_warns > total_reps * 0.3:
        recommendations.append("• Practice tracking knees over toes - may need ankle mobility work")
    
    if not recommendations:
        recommendations.append("• Great form! Continue practicing to maintain consistency")
        recommendations.append("• Consider adding weight gradually to challenge your form")
    
    for rec in recommendations:
        story.append(Paragraph(rec, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return FormAnalyzer()

analyzer = get_analyzer()

# Title
st.title("💪 RepCheck")
st.markdown("### AI-Powered Lifting Form Analysis")
st.markdown("📸 Capture your squat • 📊 Get instant feedback • 📱 Works on any device")

st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs(["📸 Photo Analysis", "🎥 Video Upload", "📊 Stats & Progress"])

# Tab 1: Photo Analysis
with tab1:
    st.subheader("Analyze Your Squat Form")
    
    with st.expander("📖 How to capture perfect squat photos", expanded=False):
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
    
    if image_to_analyze:
        st.divider()
        st.subheader("Your Image")
        
        image = Image.open(image_to_analyze)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        
        with col2:
            st.markdown("*Annotated image will appear here after analysis*")
        
        if st.button("🔍 Analyze Form", key="analyze_btn", use_container_width=True, type="primary"):
            with st.spinner("Analyzing your form..."):
                try:
                    result = analyzer.analyze_squat(image)
                    
                    if result['success']:
                        # Add to history
                        result['image'] = image.copy()
                        st.session_state.analysis_history.append(result)
                        
                        st.success("✅ Analysis Complete!")
                        
                        if result['all_pass']:
                            st.markdown('<div class="success-box"><h3>🎉 Excellent Form!</h3><p>All checks passed. Keep up the great work!</p></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="fail-box"><h3>⚠️ Form Issues Detected</h3><p>Review the feedback below to improve your form.</p></div>', unsafe_allow_html=True)
                        
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
                        st.subheader("📸 Annotated Image")
                        
                        col1_result, col2_result = st.columns([1, 1])
                        
                        with col1_result:
                            st.image(image, caption="Original", use_container_width=True)
                        
                        with col2_result:
                            annotated = analyzer.annotate_image(image, result)
                            st.image(annotated, caption="Form Analysis", use_container_width=True)
                            
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

# Tab 3: Stats & Progress
with tab3:
    st.subheader("📊 Your Stats & Progress")
    
    history = st.session_state.analysis_history
    total_reps = len(history)
    
    if total_reps == 0:
        st.info("📊 No analysis data yet. Analyze some reps to see your progress!")
        st.markdown("**Get Started:**")
        st.markdown("1. Go to the 'Photo Analysis' tab")
        st.markdown("2. Upload or capture photos of your squats")
        st.markdown("3. Analyze at least 5 reps for basic insights")
        st.markdown("4. Analyze 15+ reps for comprehensive statistics")
    
    elif total_reps < 5:
        st.warning(f"📊 You've analyzed {total_reps} rep{'s' if total_reps > 1 else ''}. Analyze at least 5 reps for meaningful insights.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reps", total_reps)
        with col2:
            passed = sum(1 for h in history if h['all_pass'])
            st.metric("Perfect Form", passed)
        with col3:
            pass_rate = (passed / total_reps * 100) if total_reps > 0 else 0
            st.metric("Pass Rate", f"{pass_rate:.0f}%")
    
    else:
        # Comprehensive stats for 5+ reps
        if total_reps < 15:
            st.info(f"📊 You've analyzed {total_reps} reps. Analyze 15+ reps for comprehensive analysis and better insights!")
        else:
            st.success(f"📊 Great! You've analyzed {total_reps} reps. Here's your comprehensive analysis:")
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        passed_reps = sum(1 for h in history if h['all_pass'])
        pass_rate = (passed_reps / total_reps * 100)
        depth_fails = sum(1 for h in history if not h['checks'].get('depth', True))
        back_fails = sum(1 for h in history if not h['checks'].get('back_angle', True))
        knee_warns = sum(1 for h in history if not h['checks'].get('knee_alignment', True))
        
        with col1:
            st.metric("Total Reps", total_reps)
        with col2:
            st.metric("Perfect Form", passed_reps)
        with col3:
            st.metric("Pass Rate", f"{pass_rate:.1f}%", 
                        delta=f"{pass_rate - 70:.1f}%" if pass_rate >= 70 else None,
                        delta_color="normal" if pass_rate >= 70 else "inverse")
        with col4:
            most_common_issue = max([
                ("Depth", depth_fails),
                ("Back Angle", back_fails),
                ("Knee Alignment", knee_warns)
            ], key=lambda x: x[1])
            st.metric("Most Common Issue", most_common_issue[0], 
                        f"{most_common_issue[1]} times")
        
        st.divider()
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Form Check Breakdown")
            
            check_data = pd.DataFrame({
                'Check': ['Depth', 'Back Angle', 'Knee Alignment'],
                'Passes': [
                    total_reps - depth_fails,
                    total_reps - back_fails,
                    total_reps - knee_warns
                ],
                'Failures': [depth_fails, back_fails, knee_warns]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Pass',
                x=check_data['Check'],
                y=check_data['Passes'],
                marker_color='#4CAF50'
            ))
            fig.add_trace(go.Bar(
                name='Fail',
                x=check_data['Check'],
                y=check_data['Failures'],
                marker_color='#F44336'
            ))
            
            fig.update_layout(
                barmode='stack',
                height=350,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            st.subheader("Overall Performance")
            
            perf_data = pd.DataFrame({
                'Category': ['Perfect Form', 'Has Issues'],
                'Count': [passed_reps, total_reps - passed_reps]
            })
            
            fig = px.pie(
                perf_data,
                values='Count',
                names='Category',
                color='Category',
                color_discrete_map={'Perfect Form': '#4CAF50', 'Has Issues': '#F44336'},
                height=350
            )
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Recommendations
        st.subheader("💡 Personalized Recommendations")
        
        recommendations = []
        if depth_fails > total_reps * 0.3:
            recommendations.append({
                'icon': '🔻',
                'title': 'Improve Depth',
                'description': f"You fail depth checks {depth_fails}/{total_reps} times ({depth_fails/total_reps*100:.0f}%). Focus on hip mobility and flexibility exercises."
            })
        if back_fails > total_reps * 0.3:
            recommendations.append({
                'icon': '🧍',
                'title': 'Fix Back Angle',
                'description': f"You fail back angle checks {back_fails}/{total_reps} times ({back_fails/total_reps*100:.0f}%). Work on core strength and keeping your chest up."
            })
        if knee_warns > total_reps * 0.3:
            recommendations.append({
                'icon': '🦵',
                'title': 'Knee Tracking',
                'description': f"Knee alignment issues detected {knee_warns}/{total_reps} times ({knee_warns/total_reps*100:.0f}%). Practice tracking knees over toes."
            })
        
        if recommendations:
            for rec in recommendations:
                st.warning(f"**{rec['icon']} {rec['title']}**\n\n{rec['description']}")
        else:
            st.success("🎉 **Excellent!** Your form is consistently good across all checks. Keep up the great work!")
        
        st.divider()
        
        # PDF Download
        st.subheader("📄 Download Progress Report")
        st.markdown("Generate a comprehensive PDF report to share with your trainer or track your progress.")
        
        if st.button("📥 Generate PDF Report", use_container_width=True, type="primary"):
            with st.spinner("Generating your report..."):
                pdf_buffer = generate_pdf_report(history)
                st.download_button(
                    label="💾 Download PDF",
                    data=pdf_buffer,
                    file_name=f"repcheck_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("✅ Report generated! Click above to download.")

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