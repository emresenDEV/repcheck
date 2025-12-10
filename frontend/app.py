import streamlit as st
import requests
from PIL import Image
import io

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

# API URL
API_URL = "http://localhost:8000"

# Title and description
st.title("💪 RepCheck")
st.markdown("### AI-Powered Lifting Form Analysis")
st.markdown("📸 Capture your squat • 📊 Get instant feedback • 📱 Works on any device")

st.divider()

# Create tabs for different modes
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
        - Use a Bluetooth remote or self-capture
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
        # Simple upload
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png', 'heic'])
        image_to_analyze = uploaded_file
        
    else:
        # Camera input
        st.info("💡 **Tip:** Use a Bluetooth remote capture hands-free!")
        camera_photo = st.camera_input("📸 Position yourself and take photo")
        image_to_analyze = camera_photo
    
    # Display and analyze image
    if image_to_analyze:
        st.divider()
        st.subheader("Your Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image_to_analyze, caption="Original", use_container_width=True)
        
        with col2:
            # Placeholder for annotated image
            st.markdown("*Annotated image will appear here after analysis*")
        
        # Analyze button
        if st.button("🔍 Analyze Form", key="analyze_btn", use_container_width=True, type="primary"):
            with st.spinner("Analyzing your form..."):
                try:
                    # Reset file pointer to beginning
                    image_to_analyze.seek(0)
                    
                    # Send to API
                    files = {"file": ("image.jpg", image_to_analyze, "image/jpeg")}
                    response = requests.post(f"{API_URL}/api/analyze", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result['success']:
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
                                # Reset and show original again
                                image_to_analyze.seek(0)
                                st.image(image_to_analyze, caption="Original", use_container_width=True)
                            
                            with col2_result:
                                annotated_url = f"{API_URL}{result['annotated_image']}"
                                st.image(annotated_url, caption="Form Analysis", use_container_width=True)
                            
                            # Download option
                            st.divider()
                            st.markdown("**💾 Save your results:**")
                            st.markdown(f"[Download Annotated Image]({annotated_url})")
                            
                        else:
                            st.error(f"❌ {result.get('error', 'Unknown error')}")
                            st.info("💡 Make sure a person is clearly visible from a **side angle**")
                            
                    elif response.status_code == 400:
                        st.error("❌ Invalid image file. Please upload JPG, PNG, or JPEG.")
                    elif response.status_code == 500:
                        st.error("❌ Server error during analysis")
                        st.info("Check the backend terminal for detailed error messages")
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("🔧 Make sure the backend server is running on port 8000")
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
    
    try:
        response = requests.get(f"{API_URL}/api/stats")
        if response.status_code == 200:
            stats = response.json()
            
            st.metric("Total Images Analyzed", stats['total_analyzed'], help="Number of images you've analyzed")
            
            # Simple progress toward goal
            st.progress(min(stats['total_analyzed'] / 50, 1.0))
            st.caption(f"Goal: Analyze 50 reps")
            
        else:
            st.error("Could not load stats")
    except Exception as e:
        st.error("📊 Stats unavailable - backend may not be running")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>RepCheck v1.0</strong> | AI-Powered Form Analysis</p>
        <p>💪 Lift safer, lift better</p>
        <p style='font-size: 12px; margin-top: 10px;'>
            Built with YOLOv8, FastAPI, and Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)


