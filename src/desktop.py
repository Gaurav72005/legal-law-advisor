import sys
import threading
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
import webview
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Import the FastAPI app from api.py
from api import app as api_app

# -----------------------------------------------------------------
# MOUNT FRONTEND
# -----------------------------------------------------------------
# We assume the React build is located in frontend/dist
frontend_dir = Path(__file__).parent.parent / "frontend" / "dist"

# To avoid errors if the directory doesn't exist yet:
if frontend_dir.exists():
    api_app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    @api_app.get("/")
    def index():
        return {"message": "Frontend build not found. Please build the React app into frontend/dist."}

# -----------------------------------------------------------------
# SERVER THREAD
# -----------------------------------------------------------------
def run_server():
    print("Starting FastAPI server...")
    uvicorn.run(api_app, host="127.0.0.1", port=8000, log_level="error")

if __name__ == "__main__":
    # Start the server in a daemon thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for the server to start (simple sleep, in production you might poll /api/status)
    print("Waiting for server to start...")
    time.sleep(2)
    
    # Open the pywebview window
    print("Opening Desktop Window...")
    webview.create_window(
        "Motor & Cyber Law Advisor",
        "http://127.0.0.1:8000",
        width=1000,
        height=800,
        min_size=(600, 480)
    )
    webview.start()
