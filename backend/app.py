from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from form_analyzer import FormAnalyzer
import shutil
from pathlib import Path
import uuid

# Initialize FastAPI app
app = FastAPI(title="RepCheck API", version="1.0.0")

# Enable CORS (allows frontend to talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize form analyzer
analyzer = FormAnalyzer()

# Create data directories if they don't exist
UPLOAD_DIR = Path("data/uploads")
PROCESSED_DIR = Path("data/processed")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "RepCheck API is running!",
        "version": "1.0.0"
    }

@app.post("/api/analyze")
async def analyze_form(file: UploadFile = File(...)):
    """
    Analyze squat form from uploaded image
    
    Args:
        file: Uploaded image file (JPG, PNG, JPEG)
        
    Returns:
        JSON with analysis results and annotated image path
    """
    # Validate file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: JPG, JPEG, PNG. Got: {file.content_type}"
        )
    
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1]
        original_filename = f"{file_id}.{file_extension}"
        original_path = UPLOAD_DIR / original_filename
        
        # Save uploaded file
        with original_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Analyzing image: {original_filename}")
        
        # Analyze form
        analysis = analyzer.analyze_squat(str(original_path))
        
        if not analysis['success']:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": analysis['error'],
                    "message": "Could not analyze form. Make sure person is visible in image."
                }
            )
        
        # Create annotated image
        annotated_filename = f"{file_id}_annotated.{file_extension}"
        annotated_path = PROCESSED_DIR / annotated_filename
        
        annotated_image = analyzer.annotate_image(str(original_path), analysis)
        annotated_image.save(str(annotated_path))
        
        print(f"Analysis complete! All checks passed: {analysis['all_pass']}")
        
        # Return results
        return {
            "success": True,
            "file_id": file_id,
            "all_pass": analysis['all_pass'],
            "feedback": analysis['feedback'],
            "num_people": analysis['num_people'],
            "original_image": f"/images/uploads/{original_filename}",
            "annotated_image": f"/images/processed/{annotated_filename}"
        }
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/images/uploads/{filename}")
async def get_upload(filename: str):
    """Serve uploaded images"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

@app.get("/images/processed/{filename}")
async def get_processed(filename: str):
    """Serve processed/annotated images"""
    file_path = PROCESSED_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

@app.get("/api/stats")
async def get_stats():
    """Get basic statistics"""
    num_uploads = len(list(UPLOAD_DIR.glob("*")))
    num_processed = len(list(PROCESSED_DIR.glob("*")))
    
    return {
        "total_analyzed": num_uploads,
        "total_processed": num_processed
    }

# Run with: uvicorn backend.app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# **What this API does:**

# **Endpoints created:**
# 1. `GET /` - Health check (is server running?)
# 2. `POST /api/analyze` - Upload image, analyze form
# 3. `GET /images/uploads/{filename}` - View uploaded images
# 4. `GET /images/processed/{filename}` - View annotated images
# 5. `GET /api/stats` - Get statistics
