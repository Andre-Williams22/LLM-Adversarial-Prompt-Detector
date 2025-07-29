import os
import time
import mlflow
import gradio as gr
# from openai import OpenAI  # Commented out for speed optimization
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge
from utils.model_processing import load_models, detect_adversarial_prompt
from utils.fast_detection import detect_adversarial_prompt_fast
from utils.mongodb_manager import mongodb_manager
import logging
import uuid

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API key setup - Commented out for speed optimization
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up MLflow
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
mlflow.set_tracking_uri(mlflow_uri)

# Set experiment with robust error handling
def setup_mlflow_experiment():
    experiment_name = "adversarial_prompt_detector"
    try:
        # Try to get existing experiment first
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Create experiment if it doesn't exist
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            print(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        return True
    except Exception as e:
        print(f"MLflow experiment setup failed: {e}")
        print("Continuing without MLflow logging...")
        return False

# Setup MLflow
mlflow_available = setup_mlflow_experiment()

# Prometheus metrics
MODEL_INFERENCE_DURATION = Histogram(
    'model_inference_duration_seconds',
    'Time spent on model inference',
    ['model_name', 'model_type']
)

MODEL_INFERENCE_COUNT = Counter(
    'model_inference_total',
    'Total number of model inferences',
    ['model_name', 'model_type']
)

ADVERSARIAL_DETECTIONS = Counter(
    'adversarial_detections_total',
    'Total number of adversarial prompts detected',
    ['model_name']
)

CHAT_REQUESTS = Counter(
    'chat_requests_total',
    'Total number of chat requests processed'
)

ACTIVE_MODELS = Gauge(
    'active_models_count',
    'Number of currently loaded models'
)

# Initialize detectors with lazy loading to speed up startup
print("🚀 Fast adversarial detection models will be loaded on first use (lazy loading)")
detectors = {"fast_ensemble": "lazy_loading"}  # Will be loaded on first request

# Global flag to track model loading status
models_loaded = False
models_loading = False

# Set the active models gauge (will be updated when models actually load)
ACTIVE_MODELS.set(0)  # Models not loaded yet

print("✅ Application ready - models will load on first detection request")

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
    """Close MongoDB connection on shutdown"""
    await mongodb_manager.close()

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
    """Endpoint to trigger model loading (warm-up)"""
    global models_loaded, models_loading
    
    if models_loaded:
        return {
            "status": "already_loaded",
            "message": "Models are already loaded and ready"
        }
    
    if models_loading:
        return {
            "status": "loading",
            "message": "Models are already loading, please wait..."
        }
    
    # Start loading in background
    import threading
    
    def load_models_background():
        ensure_models_loaded()
    
    thread = threading.Thread(target=load_models_background, daemon=True)
    thread.start()
    
    return {
        "status": "started",
        "message": "Model loading started in background",
        "estimated_time": "30-60 seconds",
        "check_status": "/loading-status"
    }

@app.get("/loading-status")
def loading_status():
    """Check model loading status"""
    global models_loaded, models_loading
    
    if models_loaded:
        return {
            "status": "ready",
            "message": "Models are loaded and ready for detection",
            "models_loaded": True,
            "models_loading": False
        }
    elif models_loading:
        return {
            "status": "loading",
            "message": "Models are currently loading, please wait...",
            "models_loaded": False,
            "models_loading": True,
            "estimated_time": "30-60 seconds"
        }
    else:
        return {
            "status": "lazy",
            "message": "Models will load on first detection request",
            "models_loaded": False,
            "models_loading": False,
            "note": "First request may take 30-60 seconds"
        }

@app.get("/health")
def health_check():
    """Enhanced health check endpoint"""
    try:
        global models_loaded, models_loading
        
        # Check actual model status
        if models_loaded:
            status = "healthy"
            model_status = "loaded"
            model_count = 4
        elif models_loading:
            status = "loading"
            model_status = "loading"
            model_count = 0
        else:
            status = "ready"
            model_status = "lazy_loading"
            model_count = 0
        
        return {
            "status": status,
            "models_loaded": models_loaded,
            "models_loading": models_loading,
            "model_status": model_status,
            "model_count": model_count,
            "timestamp": time.time(),
            "version": "1.0.0",
            "startup_mode": "lazy_loading",
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
            "models_loaded": models_loaded,
            "models_loading": models_loading,
            "startup_mode": "lazy_loading",
            "model_count": 4 if models_loaded else 0
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
    """Ensure models are loaded (lazy loading on first use)"""
    global models_loaded, models_loading
    
    if models_loaded:
        return True
    
    if models_loading:
        # Wait for models to finish loading (up to 60 seconds)
        wait_count = 0
        while models_loading and wait_count < 60:
            time.sleep(1)
            wait_count += 1
        return models_loaded
    
    # Start loading models
    models_loading = True
    try:
        print("🔄 Loading ML models (first request - this may take 30-60 seconds)...")
        start_time = time.time()
        
        # Import and test the fast detection system
        from utils.fast_detection import detect_adversarial_prompt_fast
        
        # Test with a simple message to ensure models load
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # This will trigger model loading in the fast detection system
        test_result = loop.run_until_complete(detect_adversarial_prompt_fast("Hello"))
        
        loop.close()
        
        load_duration = time.time() - start_time
        
        models_loaded = True
        models_loading = False
        
        # Update metrics
        ACTIVE_MODELS.set(4)
        detectors["fast_ensemble"] = "loaded"
        
        print(f"✅ ML models loaded successfully in {load_duration:.2f} seconds")
        return True
        
    except Exception as e:
        models_loading = False
        print(f"❌ Failed to load models: {e}")
        logger.error(f"Model loading failed: {e}")
        return False

def chat_and_detect(user_message, history):
    """
    Chat function with robust error handling and asyncio compatibility
    Returns: (history, history, flag_note) tuple for Gradio interface
    """
    start_time = time.time()
    history = history or []
    history.append(("User", user_message))
    
    # Initialize defaults
    bot_response = "An error occurred while processing your request."
    flag_note = "<p style='color:red;'>Processing error</p>"
    is_adv = False
    reasoning = {"reason": "Unknown", "scores": [0.0, 0.0, 0.0, 0.0], "threshold": 0.5}
    
    # Generate session ID for this conversation
    session_id = str(uuid.uuid4())

    # Track chat request
    CHAT_REQUESTS.inc()

    try:
        # 1. Ensure models are loaded (lazy loading)
        if not ensure_models_loaded():
            # If models failed to load, use fallback detection immediately
            logger.warning("Models failed to load, using fallback detection")
            user_message_lower = user_message.lower()
            basic_adversarial_keywords = [
                "ignore", "jailbreak", "bypass", "override", "previous instructions",
                "act as", "pretend", "roleplay", "system prompt", "admin", "root",
                "prompt injection", "escape", "break out", "sudo", "developer mode"
            ]
            
            keyword_detected = any(keyword in user_message_lower for keyword in basic_adversarial_keywords)
            is_adv, reasoning = keyword_detected, {
                "reason": f"Fallback detection - {'Basic keyword match' if keyword_detected else 'No obvious patterns'}: Models failed to load",
                "scores": [0.8 if keyword_detected else 0.1, 0.0, 0.0, 0.0],
                "threshold": 0.5
            }
        else:
            # 2. Fast adversarial prompt detection (models are now loaded)
            detection_start = time.time()
            
            # Use sync wrapper for the async detection function
            try:
                import asyncio
                import threading
                
                # Store result in a thread-safe way
                detection_result = {"result": None, "error": None}
                
                def detection_thread():
                    """Background thread for async detection"""
                    try:
                        # Create new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Run the async detection
                        result = loop.run_until_complete(detect_adversarial_prompt_fast(user_message))
                        detection_result["result"] = result
                        
                        loop.close()
                        
                    except Exception as e:
                        detection_result["error"] = str(e)
                
                # Run detection in background thread and wait for completion
                thread = threading.Thread(target=detection_thread)
                thread.start()
                thread.join(timeout=60)  # Increase timeout to 60 seconds
                
                if detection_result["error"]:
                    raise Exception(detection_result["error"])
                elif detection_result["result"]:
                    is_adv, reasoning = detection_result["result"]
                else:
                    # Log more details about the timeout
                    logger.error(f"Detection thread timeout after 60 seconds for message: {user_message[:100]}...")
                    raise Exception("Detection thread timeout after 60 seconds")
                    
            except Exception as detection_error:
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
                
                logger.info(f"Using fallback detection: {is_adv} - {reasoning['reason']}")
            
            detection_duration = time.time() - detection_start
        
        # Handle fallback case (when models failed to load)
        if not models_loaded:
            detection_duration = 0.001  # Very fast fallback detection
        
        # Track model metrics for fast ensemble
        model_names = ["keyword_detector", "toxic_bert", "hate_roberta", "safety_bart"]
        scores_list = reasoning["scores"]
        
        for i, score in enumerate(scores_list):
            if i < len(model_names):
                model_name = model_names[i]
                MODEL_INFERENCE_COUNT.labels(model_name=model_name, model_type="fast_detector").inc()
                MODEL_INFERENCE_DURATION.labels(model_name=model_name, model_type="fast_detector").observe(detection_duration)
                
                # Track if adversarial prompt was detected by this model
                if score > reasoning["threshold"]:
                    ADVERSARIAL_DETECTIONS.labels(model_name=model_name).inc()
        
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
        bot_response = "An error occurred while processing your request."
        history.append(("Bot", bot_response))
        flag_note = f"<p style='color:orange;font-weight:bold;'>Error: {str(e)}</p>"

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
                import asyncio
                import time
                
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
        import threading
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
            <p style="color:#6c757d;margin:0;">
                <strong>Note:</strong> The first message may take 30-60 seconds as ML models load. 
                Subsequent responses will be much faster. 
                <a href="/loading-status" target="_blank" style="color:#007BFF;">Check loading status</a>
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
    import uvicorn
    # Use port 80 to match Dockerfile, single worker for better resource usage
    uvicorn.run(app, host="0.0.0.0", port=80, workers=1)
