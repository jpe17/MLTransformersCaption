#!/usr/bin/env python3
"""
Startup script for the Image Captioning FastAPI app
"""

import uvicorn
import os
import sys

def main():
    # Add current directory to path so we can import modules
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("🚀 Starting Image Captioning Web App...")
    print("📍 Make sure you have the model file at: saved_models/explained/explained_20_epochs.pth")
    print("🌐 The app will be available at: http://localhost:8000")
    print("📱 Or on your network at: http://0.0.0.0:8000")
    print("-" * 60)
    
    # Run the app
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

if __name__ == "__main__":
    main() 