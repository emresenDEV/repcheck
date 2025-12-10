from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from form_analyzer import FormAnalyzer
import shutil
from pathlib import Path
import uuid
from PIL import Image
import io

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
        file: Uploaded image file (JPG, PNG, JPEG, WEBP)
        
    Returns:
        JSON with analysis results and annotated image path
    """
    try:
        # Read file content
        content = await file.read()
        
        # Try to open as image (validates it's actually an image)
        try:
            image = Image.open(io.BytesIO(content))
            image.verify()  # Verify it's a valid image
            
            # Re-open for processing (verify() closes the file)
            image = Image.open(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file. Please upload JPG, PNG, or JPEG. Error: {str(e)}"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        
        # Determine extension from image format
        if image.format:
            extension = image.format.lower()
        else:
            extension = 'jpg'  # Default
        
        original_filename = f"{file_id}.{extension}"
        original_path = UPLOAD_DIR / original_filename
        
        # Save uploaded file
        image.save(str(original_path))
        
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
        annotated_filename = f"{file_id}_annotated.{extension}"
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
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
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

