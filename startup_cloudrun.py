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
    import gradio as gr
    
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
    
    # Import and mount the full app after server starts
    def mount_full_app():
        """Mount the full application after models are ready"""
        try:
            # Wait for models to be ready or fail
            timeout = 300  # 5 minutes max
            start_time = time.time()
            
            while models_loading and (time.time() - start_time) < timeout:
                time.sleep(1)
            
            if models_ready:
                logger.info("🔄 Mounting full application with Gradio interface...")
                
                # Import the main application
                from main import demo
                
                # Mount Gradio interface
                app = gr.mount_gradio_app(app, demo, path="/chat")
                logger.info("✅ Full application mounted successfully!")
                
            else:
                logger.error("❌ Failed to mount full application - models not ready")
                
        except Exception as e:
            logger.error(f"❌ Error mounting full application: {e}")
    
    # Start full app mounting in background
    mount_thread = threading.Thread(target=mount_full_app, daemon=True)
    mount_thread.start()
    
    return app

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment
    port = int(os.getenv("PORT", 8080))
    
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
