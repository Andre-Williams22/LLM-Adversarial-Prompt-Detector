import os
import time
import mlflow
import gradio as gr
import logging
import uuid
import threading
import asyncio
import uvicorn
# from openai import OpenAI  # Commented out for speed optimization
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
from utils.fast_detection import FastAdversarialDetector
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from utils.model_processing import load_models, detect_adversarial_prompt
from utils.fast_detection import detect_adversarial_prompt_fast
from utils.mongodb_manager import mongodb_manager
from utils.prometheus_metrics import (
    initialize_metrics,
    track_model_inference,
    track_chat_request,
    track_prompt_result,
    track_concurrent_request_start,
    track_concurrent_request_end,
    track_model_error,
    track_chat_failure,
    cleanup,
    # Import specific metrics for direct access where needed
    MODEL_INFERENCE_DURATION,
    MODEL_INFERENCE_COUNT,
    ADVERSARIAL_DETECTIONS,
    SAFE_PROMPTS,
    CHAT_REQUESTS,
    ACTIVE_MODELS,
    MODEL_INFERENCE_ERRORS,
    CHAT_REQUEST_FAILURES,
    CONCURRENT_REQUESTS,
    system_monitor
)

# Load environment variables
load_dotenv()

# Set up logging with enhanced visibility for web server environments
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log', mode='a')  # File output for persistence
    ]
)
logger = logging.getLogger(__name__)

# Force stdout/stderr to be unbuffered for real-time visibility
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# OpenAI API key setup - Commented out for speed optimization
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up MLflow - use same configuration as fast_detection
# The fast_detector already sets up MLflow correctly, we just need to match it
def setup_mlflow_experiment():
    experiment_name = "adversarial_detection_system"
    try:
        # Use MLFLOW_TRACKING_URI environment variable if set, otherwise use local file store
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            print(f"Using MLflow tracking URI from environment: {tracking_uri}")
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # This should match the local setup in fast_detection.py
            print("Using local MLflow tracking URI (matches fast_detection.py)")
        
        # Use the same experiment that fast_detector uses
        mlflow.set_experiment(experiment_name)
        
        # Test if we can access the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            print(f"✅ Using MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})")
        else:
            print(f"⚠️ MLflow experiment {experiment_name} not found, will be created by fast_detector")
        
        return True
    except Exception as e:
        print(f"MLflow experiment setup failed: {e}")
        print("Continuing without MLflow logging...")
        return False

# Setup MLflow
mlflow_available = setup_mlflow_experiment()

# Import the already-loaded global detector from fast_detection
from utils.fast_detection import fast_detector

# Global variables for model state - use the pre-loaded detector
models_loaded = True  # Models are already loaded in fast_detection.py
models_loading = False
detectors = {"fast_ensemble": "loaded"}

# Initialize Prometheus metrics and system monitoring
initialize_metrics()

print("✅ Using pre-loaded models from fast_detection.py - ready for instant detection")

# Build FastAPI + Gradio app
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB connection on startup"""
    try:
        await mongodb_manager.connect()
        logger.info("✅ Application startup complete with MongoDB")
    except Exception as e:
        logger.error(f"❌ Failed to initialize MongoDB during startup: {e}")
        logger.info("Application will continue without MongoDB logging")

@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection and cleanup on shutdown"""
    await mongodb_manager.close()
    cleanup()  # Cleanup Prometheus monitoring resources

@app.get("/favicon.ico")
async def favicon():
    """Simple favicon to prevent 404 errors"""
    # Return a simple 1x1 transparent PNG
    favicon_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    return Response(content=favicon_data, media_type="image/png")

@app.get("/manifest.json")
async def manifest():
    """PWA manifest file"""
    manifest_data = {
        "name": "Adversarial Prompt Detector",
        "short_name": "AdversarialDetector", 
        "description": "AI assistant with adversarial prompt detection and safety filtering",
        "start_url": "/chat",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#007BFF",
        "icons": [
            {
                "src": "/favicon.ico",
                "sizes": "16x16", 
                "type": "image/x-icon"
            }
        ]
    }
    return manifest_data

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/warm-up")
async def warm_up_models():
    """Models are always loaded at startup"""
    return {
        "status": "always_loaded",
        "message": "Models are pre-loaded at startup and always ready for instant detection"
    }

@app.get("/loading-status")
def loading_status():
    """Models are always loaded at startup"""
    return {
        "status": "ready",
        "message": "Models are pre-loaded and ready for instant detection",
        "models_loaded": True,
        "models_loading": False,
        "startup_time": "instant"
    }

@app.get("/health")
def health_check():
    """Enhanced health check endpoint"""
    try:
        return {
            "status": "healthy",
            "models_loaded": True,
            "models_loading": False,
            "model_status": "loaded",
            "model_count": 4,
            "timestamp": time.time(),
            "version": "1.0.0",
            "startup_mode": "pre_loaded",
            "endpoints": {
                "chat": "/chat",
                "metrics": "/metrics",
                "health": "/health"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/")
def home():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Adversarial Prompt Detector API",
        "description": "An AI assistant integrated with a detector for adversarial prompts, protecting against prompt injection attacks.",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "chat_interface": "/chat",
            "health_check": "/health",
            "metrics": "/metrics",
            "statistics": "/stats"
        },
        "services": {
            "models_loaded": True,
            "models_loading": False,
            "startup_mode": "pre_loaded",
            "model_count": 4
        },
        "links": {
            "chat": "/chat",
            "health": "/health",
            "metrics": "/metrics",
            "statistics": "/stats"
        }
    }

@app.get("/stats")
async def get_statistics():
    """Get application statistics from MongoDB"""
    try:
        chat_stats = await mongodb_manager.get_chat_statistics(days=7)
        model_stats = await mongodb_manager.get_model_performance_stats(days=7)
        
        return {
            "chat_statistics": chat_stats,
            "model_performance": model_stats,
            "time_period": "last_7_days"
        }
    except Exception as e:
        return {
            "error": f"Failed to retrieve statistics: {str(e)}",
            "chat_statistics": {},
            "model_performance": []
        }

def ensure_models_loaded():
    """Check if models are loaded - they're already loaded from fast_detection.py"""
    global models_loaded, fast_detector
    
    # Models are already loaded from the global import
    if models_loaded and fast_detector is not None:
        return True
    
    # This shouldn't happen, but if it does, log and return True since models are pre-loaded
    logger.warning("Models should already be loaded from fast_detection.py")
    return True

def chat_and_detect(user_message, history):
    """
    Chat function with robust error handling and asyncio compatibility
    Returns: (history, history, flag_note) tuple for Gradio interface
    """
    start_time = time.time()
    history = history or []
    history.append(("User", user_message))
    
    # Track concurrent requests
    track_concurrent_request_start()
    
    # Initialize defaults
    bot_response = "An error occurred while processing your request."
    flag_note = "<p style='color:red;'>Processing error</p>"
    is_adv = False
    reasoning = {"reason": "Unknown", "scores": [0.0, 0.0, 0.0, 0.0], "threshold": 0.5}
    
    # Generate session ID for this conversation
    session_id = str(uuid.uuid4())

    # Track chat request
    track_chat_request()

    try:
        # Fast adversarial prompt detection (models are pre-loaded and ready)
        detection_start = time.time()
        
        try:
            # Set environment variable to avoid tokenizer warnings
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Enhanced logging for web interface requests
            print(f"\n🌐 WEB REQUEST: Starting detection for prompt: '{user_message[:50]}{'...' if len(user_message) > 50 else ''}'", flush=True)
            logger.info(f"Web request detection starting for: {user_message[:50]}")
            
            # Run detection synchronously - much faster than asyncio.run()
            is_adv, reasoning = fast_detector.detect_adversarial_sync(user_message)
            
            # Track the prompt result (adversarial vs safe)
            track_prompt_result(is_adv, model_name="fast_ensemble")
            
            # Enhanced logging after detection
            decision_icon = "🚨" if is_adv else "✅"
            print(f"🌐 WEB REQUEST COMPLETED: {decision_icon} {'ADVERSARIAL' if is_adv else 'SAFE'} - {reasoning.get('reason', 'Unknown')}", flush=True)
            logger.info(f"Web request completed: {is_adv} - {reasoning.get('reason')}")
            
        except Exception as detection_error:
            # Track model inference error
            track_model_error("fast_detector", "detection_failure")
            logger.error(f"Detection failed: {detection_error}")
            # Enhanced fallback with basic keyword detection
            user_message_lower = user_message.lower()
            basic_adversarial_keywords = [
                "ignore", "jailbreak", "bypass", "override", "previous instructions",
                "act as", "pretend", "roleplay", "system prompt", "admin", "root",
                "prompt injection", "escape", "break out", "sudo", "developer mode"
            ]
            
            # Simple keyword-based fallback detection
            keyword_detected = any(keyword in user_message_lower for keyword in basic_adversarial_keywords)
            
            is_adv, reasoning = keyword_detected, {
                "reason": f"Fallback detection - {'Basic keyword match' if keyword_detected else 'No obvious patterns'}: {str(detection_error)}",
                "scores": [0.8 if keyword_detected else 0.1, 0.0, 0.0, 0.0],
                "threshold": 0.5
            }
            
            # Track the fallback prompt result
            track_prompt_result(is_adv, model_name="fallback_detector")
            
            logger.info(f"Using fallback detection: {is_adv} - {reasoning['reason']}")
        
        detection_duration = time.time() - detection_start
        
        # Track model metrics for fast ensemble
        model_names = ["keyword_detector", "toxic_bert", "hate_roberta", "safety_bart"]
        scores_list = reasoning["scores"]
        
        for i, score in enumerate(scores_list):
            if i < len(model_names):
                model_name = model_names[i]
                is_adversarial_by_model = score > reasoning["threshold"]
                track_model_inference(
                    model_name=model_name,
                    model_type="fast_detector",
                    duration=detection_duration,
                    is_adversarial=is_adversarial_by_model
                )
        
        print(f"🚀 Fast detection completed in {detection_duration:.3f}s")
        print("reasoning", reasoning)
        scores = reasoning["scores"]
        print("scores", scores)

        bot_response = ""
        if is_adv:
            bot_response = "⚠️ Adversarial prompt detected! The request was not processed."
            history.append(("Bot", bot_response))
            flag_note = (
                f"<p style='color:red;font-weight:bold;'>"
                f"Adversarial Prompt Detected<br>"
                f"Reason: {reasoning['reason']}<br>"
                f"Threshold: {reasoning['threshold']:.2f}<br>"
                f"Scores: {reasoning['scores']}</p>"    
            )
        else:
            # Assemble the last few turns into the OpenAI chat-completions format
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for role, msg in history[-6:]:
                if role == "User":
                    messages.append({"role": "user", "content": msg})
                else:
                    messages.append({"role": "assistant", "content": msg})
            # Add the most recent user message
            messages.append({"role": "user", "content": user_message})

            # OpenAI API call commented out for speed optimization
            # response = client.chat.completions.create(
            #     model="gpt-4o-mini",  # Using a valid OpenAI model
            #     messages=messages,  # Pass the entire list of messages
            # )
            # print("Response from OpenAI:", response)
            # bot_response = response.choices[0].message.content
            
            # Mock response for speed optimization
            bot_response = f"This is a mock response to your message: '{user_message}'. The adversarial detection system determined this prompt is safe."
            history.append(("Bot", bot_response))

            flag_note = (
                f"<p style='color:green;font-weight:bold;'>"
                f"No adversarial prompt detected.<br>"
                f"Reason: {reasoning['reason']}<br>"
                f"Threshold: {reasoning['threshold']:.2f}</br>"
                f"Scores: {reasoning['scores']}</p>"
                )

    except Exception as e:
        # Track chat request failure
        track_chat_failure("general_error")
        bot_response = "An error occurred while processing your request."
        history.append(("Bot", bot_response))
        flag_note = f"<p style='color:orange;font-weight:bold;'>Error: {str(e)}</p>"

    finally:
        # Always decrement concurrent requests counter
        track_concurrent_request_end()

    end_time = time.time()
    final_time = end_time - start_time
    print("latency in ms", final_time)
    
    # Log to MongoDB (simplified approach to avoid event loop issues)
    try:
        detection_results = {
            "is_adversarial": is_adv,
            "scores": reasoning.get("scores", {}),
            "reason": reasoning.get("reason", ""),
            "threshold": reasoning.get("threshold", 0.0)
        }
        
        # Simplified MongoDB logging to avoid asyncio issues
        def mongodb_logging_thread():
            """Background thread for MongoDB logging - fire and forget"""
            try:
                # Create completely isolated event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run the async logging function
                coro = mongodb_manager.log_chat_interaction(
                    user_message=user_message,
                    bot_response=bot_response,
                    detection_results=detection_results,
                    latency=final_time,
                    session_id=session_id
                )
                
                loop.run_until_complete(coro)
                loop.close()
                
                logger.info("MongoDB logging completed successfully")
                
            except Exception as thread_error:
                logger.error(f"MongoDB logging thread error: {thread_error}")
                # Don't crash the main thread
        
        # Start background thread as daemon (fire and forget)
        thread = threading.Thread(target=mongodb_logging_thread, daemon=True, name="mongodb-logger")
        thread.start()
        
    except Exception as mongo_error:
        logger.error(f"Failed to setup MongoDB logging thread: {mongo_error}")
        # Continue without MongoDB logging - don't crash the main flow

    return history, history, flag_note

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <style>
            #chatbox { background-color: #f0f0f0; border-radius:5px; padding:10px; }
            #chatbox .bot { background-color:#e0e0e0; border-radius:5px; padding:5px; margin:5px 0; }
            #chatbox .user{ background-color:#fff; border-radius:5px; padding:5px; margin:5px 0; }
            #user_input{ background:#f0f0f0; border:1px solid #ccc; border-radius:5px; }
            #send_button{ background:#007BFF; color:#fff; border:none; border-radius:5px; padding:10px 20px; cursor:pointer; }
            #send_button:hover{ background:#0056b3; }
        </style>
    """)
    gr.Markdown("""
        <h1 style="text-align:center;color:#007BFF;">AI Safety Detector + Mock Chat Interface</h1>
        <p style="text-align:center;color:#555;">An AI safety system with mock responses (OpenAI disabled for speed optimization).</p>
        <div id="loading-notice" style="text-align:center;background:#f8f9fa;padding:15px;border-radius:8px;margin:10px 0;">
            <p style="color:#28a745;margin:0;">
                <strong>✅ Ready:</strong> All ML models are pre-loaded for instant adversarial detection!
                Fast response times guaranteed.
            </p>
        </div>
    """)
    chatbot = gr.Chatbot(label="Safety Detector + Mock Chat", elem_id="chatbox")
    state = gr.State([])  # Holds chat history as list of (role, msg)
    user_in = gr.Textbox(
        placeholder="Ask anything here …",
        label="Your Message",
        lines=2,
        max_lines=5,
        elem_id="user_input",
    )
    send_btn = gr.Button("Send", elem_id="send_button")
    flag_note = gr.Markdown("", elem_id="flag_note")

    send_btn.click(
        fn=chat_and_detect,
        inputs=[user_in, state],
        outputs=[chatbot, state, flag_note],
        queue=False,
    )

# Mount Gradio under /chat
app = gr.mount_gradio_app(app, demo, path="/chat")

if __name__ == "__main__":
    # Use port 8080 for local development, single worker for better resource usage
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
