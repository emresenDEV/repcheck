# 💪 RepCheck

AI-powered squat form analysis that helps you prevent injuries and improve technique.

[Live Demo](https://repcheck.streamlit.app) | Built with YOLOv8 & Streamlit

## What It Does

RepCheck analyzes photos of your squat form and checks three critical aspects:

- **Depth** - Are your hips below parallel?
- **Back Angle** - Is your torso upright enough?
- **Knee Alignment** - Are your knees tracking properly over your feet?

You get instant visual feedback with color-coded annotations showing exactly what needs improvement.

## Why I Built This

Improper lifting form causes thousands of preventable injuries every year. Personal trainers aren't always accessible, and watching yourself in the mirror doesn't always catch form issues. I wanted a tool that could give objective feedback on demand, from anywhere.

This was also a learning project - I wanted to understand how computer vision works in practice and build something that could actually be useful.

## How to Use It

### Quick Start (Try the Live Demo)

1. Go to [repcheck.streamlit.app](https://repcheck.streamlit.app) (note: the app will sleep after 7 days of inactivity)
2. Upload a photo or use your phone's camera
3. Make sure the photo shows your **full body from the side**
4. Click "Analyze Form"
5. Review feedback and download annotated image

### Setup Tips

For best results:

- Position camera at hip height, 3-4 feet away
- **Side angle is critical** (perpendicular to your body)
- Good lighting helps
- Use a Bluetooth remote or self-timer to capture yourself mid-squat

## Running Locally

If you want to run this on your own machine:

```bash
# Clone the repo
git clone https://github.com/emresenDEV/repcheck.git
cd repcheck

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## How It Works

RepCheck uses **YOLOv8-pose** to detect 17 keypoints on your body (shoulders, hips, knees, ankles, etc.). Then it applies some basic geometry:

**Depth Check:**

- Compares hip y-coordinate to knee y-coordinate
- Hip below knee = good depth

**Back Angle Check:**

- Calculates angle between shoulder-hip-knee points
- 45-90° = upright posture
- Less than 45° = leaning too far forward

**Knee Alignment:**

- Measures horizontal distance between knee and ankle
- Small distance = knees tracking well over feet
- Large distance = knees caving in or pushing too far out

The model runs entirely in-browser (via Streamlit Cloud), no data is stored.

## Tech Stack

- **YOLOv8** - Pose estimation (detects body keypoints)
- **OpenCV** - Image annotation (draws the colored skeleton)
- **Streamlit** - Web interface
- **NumPy** - Angle calculations

Everything runs on CPU, no GPU needed. Inference takes about 100-200ms per image.

## Limitations

- Only analyzes static images (video support coming eventually)
- Only checks squat form (could expand to deadlifts, bench press, etc.)
- Requires clear side-angle view
- Can only analyze one person at a time
- Form rules are simplified - not a replacement for a real coach

## What's Next

Some ideas I'm considering:

- Video analysis (frame-by-frame tracking)
- Rep counting
- Support for other exercises
- Progress tracking over time
- Comparison mode (upload two photos side-by-side)

Built by [@emresenDEV](https://github.com/emresenDEV)

*RepCheck v1.0 - Lift safer, lift better* 💪
