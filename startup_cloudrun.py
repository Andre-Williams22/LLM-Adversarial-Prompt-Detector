"""
Cloud Run optimized startup script
Ensures fast server startup with background model loading
"""

import os
import sys
import time
import threading
import logging
from contextlib import asynccontextmanager

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Global state for model loading
models_ready = False
models_loading = True
startup_error = None

def load_models_background():
    """Load models in background thread to not block server startup"""
    global models_ready, models_loading, startup_error
    
    try:
        logger.info("🚀 Starting background model loading...")
        
        # Import and initialize the fast detector
        from utils.fast_detection import fast_detector
        
        # Test model functionality
        test_result = fast_detector.detect_adversarial_sync("test prompt")
        
        models_ready = True
        models_loading = False
        logger.info("✅ All models loaded successfully!")
        
    except Exception as e:
        startup_error = str(e)
        models_loading = False
        models_ready = False
        logger.error(f"❌ Model loading failed: {e}")

def create_app():
    """Create FastAPI app with optimized startup"""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import gradio as gr
    
    # Request model for the detect endpoint
    class DetectRequest(BaseModel):
        text: str
    
    # Create app instance
    app = FastAPI(title="Adversarial Prompt Detector")
    
    # Start model loading in background immediately
    model_thread = threading.Thread(target=load_models_background, daemon=True)
    model_thread.start()
    
    @app.get("/")
    def root():
        return {
            "message": "Adversarial Prompt Detector API",
            "status": "running",
            "models_ready": models_ready,
            "models_loading": models_loading,
            "startup_error": startup_error
        }
    
    @app.get("/health")
    def health():
        """Health check endpoint - always responds quickly"""
        return {
            "status": "healthy" if models_ready else "starting",
            "models_loaded": models_ready,
            "models_loading": models_loading,
            "timestamp": time.time(),
            "error": startup_error
        }
    
    @app.get("/ready")
    def readiness():
        """Readiness check - only healthy when models are loaded"""
        if models_ready:
            return {"status": "ready"}
        elif startup_error:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "error": startup_error}
            )
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "loading", "message": "Models still loading..."}
            )
    
    # Add a simple detection endpoint that works immediately
    @app.post("/detect")
    def detect_simple(request: DetectRequest):
        """Simple detection endpoint"""
        if not models_ready:
            return JSONResponse(
                status_code=503,
                content={"status": "loading", "message": "Models still loading..."}
            )
        
        try:
            from utils.fast_detection import fast_detector
            result = fast_detector.detect_adversarial_sync(request.text)
            return {"result": result, "text": request.text}
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
    
    # Mount Gradio interface immediately but it will show loading until models are ready
    try:
        logger.info("🔄 Setting up Gradio interface...")
        from main import demo
        app = gr.mount_gradio_app(app, demo, path="/chat")
        logger.info("✅ Gradio interface mounted!")
    except Exception as e:
        logger.error(f"❌ Error mounting Gradio: {e}")
    
    return app

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment
    port = int(os.getenv("PORT", 80))
    
    logger.info(f"🚀 Starting server on port {port}...")
    logger.info("📋 Server will start immediately, models loading in background")
    
    # Create optimized app
    app = create_app()
    
    # Run with optimized settings for Cloud Run
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        access_log=True,
        log_level="info"
    )
