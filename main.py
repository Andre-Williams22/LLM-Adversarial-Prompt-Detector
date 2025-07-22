import os
import time
import torch
import mlflow
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge
from utils.model_processing import load_models, detect_adversarial_prompt
from utils.mongodb_manager import mongodb_manager
import logging
import uuid

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API key setup
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key or openai_api_key == "your_openai_api_key_here":
    logger.warning("OpenAI API key not configured. Chat functionality will be limited.")
    client = None
else:
    client = OpenAI(api_key=openai_api_key)

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

# Initialize detectors globally with progress logging
print("🤖 Loading adversarial detection models...")
start_load_time = time.time()
detectors = load_models()
load_duration = time.time() - start_load_time

# Set the active models gauge
ACTIVE_MODELS.set(len(detectors))

print(f"✅ Models loaded successfully in {load_duration:.2f} seconds")

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

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health_check():
    """Enhanced health check endpoint"""
    try:
        models_status = bool(detectors)
        model_count = len(detectors) if detectors else 0
        
        # Check service configurations (without exposing secrets)
        services_config = {
            "openai_configured": bool(client),
            "mongodb_configured": hasattr(mongodb_manager, 'connection_string') and mongodb_manager.connection_string is not None,
            "mlflow_configured": bool(mlflow_available)
        }
        
        return {
            "status": "healthy" if models_status else "unhealthy",
            "models_loaded": models_status,
            "model_count": model_count,
            "services": services_config,
            "timestamp": time.time(),
            "version": "1.0.0",
            "endpoints": {
                "chat": "/chat",
                "metrics": "/metrics",
                "health": "/health",
                "stats": "/stats"
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
            "models_loaded": bool(detectors),
            "model_count": len(detectors) if detectors else 0
        },
        "links": {
            "chat": "https://safe-prompts.andrewilliams.ai/chat",
            "health": "https://safe-prompts.andrewilliams.ai/health",
            "metrics": "https://safe-prompts.andrewilliams.ai/metrics",
            "statistics": "https://safe-prompts.andrewilliams.ai/stats"
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

def chat_and_detect(user_message, history):
    start_time = time.time()
    history = history or []
    history.append(("User", user_message))
    
    # Generate session ID for this conversation
    session_id = str(uuid.uuid4())

    # Track chat request
    CHAT_REQUESTS.inc()

    try:
        # 1. Check adversarial prompt
        detection_start = time.time()
        is_adv, reasoning = detect_adversarial_prompt(user_message, detectors)
        detection_duration = time.time() - detection_start
        
        # Track model metrics
        model_names = ["electra_small", "tox_bert", "offensive_roberta", "bart_mnli"]
        scores_list = reasoning["scores"]
        
        for i, score in enumerate(scores_list):
            if i < len(model_names):
                model_name = model_names[i]
                MODEL_INFERENCE_COUNT.labels(model_name=model_name, model_type="adversarial_detector").inc()
                MODEL_INFERENCE_DURATION.labels(model_name=model_name, model_type="adversarial_detector").observe(detection_duration)
                
                # Track if adversarial prompt was detected by this model
                if score > reasoning["threshold"]:
                    ADVERSARIAL_DETECTIONS.labels(model_name=model_name).inc()
        
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

            # Call the OpenAI ChatGPT API
            if client:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Using a valid OpenAI model
                    messages=messages,  # Pass the entire list of messages
                )
                print("Response from OpenAI:", response)
                bot_response = response.choices[0].message.content
                history.append(("Bot", bot_response))
            else:
                bot_response = "OpenAI API not configured. Chat functionality is disabled."
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
    
    # Log to MongoDB (async, non-blocking)
    try:
        detection_results = {
            "is_adversarial": is_adv,
            "scores": reasoning.get("scores", []),  # Changed from {} to []
            "reason": reasoning.get("reason", ""),
            "threshold": reasoning.get("threshold", 0.0)
        }
        
        # Run MongoDB logging in background
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            mongodb_manager.log_chat_interaction(
                user_message=user_message,
                bot_response=bot_response,
                detection_results=detection_results,
                latency=final_time,
                session_id=session_id
            )
        )
        loop.close()
    except Exception as mongo_error:
        logger.error(f"Failed to log to MongoDB: {mongo_error}")
        # Continue without MongoDB logging

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
        <h1 style="text-align:center;color:#007BFF;">ChatGPT + Adversarial Prompt Detector</h1>
        <p style="text-align:center;color:#555;">An AI assistant integrated with a detector for adversarial prompts.</p>
    """)
    chatbot = gr.Chatbot(label="ChatGPT + Detector", elem_id="chatbox")
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
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True, workers=1)
