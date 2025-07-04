"""
FastAPI App for Image Captioning
================================
Uses the ImageCaptioner class from inference.py
"""

import torch
from PIL import Image
import os
import io
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import your inference class
from inference import ImageCaptioner

# --- App Setup ---
app = FastAPI()

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up templates and static files with absolute paths
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "frontend", "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend", "static")), name="static")

# --- Global instances ---
CAPTIONER = None

@app.on_event("startup")
async def startup_event():
    """Load the image captioner on startup."""
    global CAPTIONER
    
    try:
        print("🚀 Loading image captioning model...")
        CAPTIONER = ImageCaptioner()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("⚠️  The app will run but captioning will fail.")

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/caption", response_class=JSONResponse)
async def caption_image(file: UploadFile = File(...)):
    """
    Takes an image file and returns a generated caption using the ImageCaptioner.
    """
    if CAPTIONER is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Generate caption using your inference class
        caption = CAPTIONER.generate_caption_from_pil(image, max_length=30)
        
        return {"caption": caption}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating caption: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating caption: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": CAPTIONER is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)