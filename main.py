import os
import time
import torch
import mlflow
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from utils.model_processing import load_models, detect_adversarial_prompt

# Load environment variables
load_dotenv()

# OpenAI API key setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
mlflow.set_experiment("adversarial_prompt_detector")

# Initialize detectors globally
detectors = load_models()

# Build FastAPI + Gradio app
app = FastAPI()

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health_check():
    detectors = load_models()
    return {"status": "healthy", "models_loaded": bool(detectors)}

@app.get("/")
def home():
    return {"message": "Welcome to the ChatGPT + Adversarial Prompt Detector API!"}

def chat_and_detect(user_message, history):
    start_time = time.time()
    history = history or []
    history.append(("User", user_message))

    try:
        # 1. Check adversarial prompt
        is_adv, reasoning = detect_adversarial_prompt(user_message, detectors)
        print("reasoning", reasoning)
        scores = reasoning["scores"]
        print("scores", scores)

        if is_adv:
            history.append(("Bot", "⚠️ Adversarial prompt detected! The request was not processed."))
            flag_note = (
                f"<p style='color:red;font-weight:bold;'>"
                f"Adversarial Prompt Detected<br>"
                f"Reason: {reasoning['reason']}<br>"
                f"Threshold: {reasoning['threshold']:.2f}<br>"
                f"Scores: {reasoning['scores']}</p>"
                
            )
            return history, history, flag_note

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
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,  # Pass the entire list of messages
        )
        print("Response from OpenAI:", response)
        bot_reply = response.choices[0].message.content
        history.append(("Bot", bot_reply))

        flag_note = (
            f"<p style='color:green;font-weight:bold;'>"
            f"No adversarial prompt detected.<br>"
            f"Reason: {reasoning['reason']}<br>"
            f"Threshold: {reasoning['threshold']:.2f}</br>"
            f"Scores: {reasoning['scores']}</p>"
            )

    except Exception as e:
        history.append(("Bot", "An error occurred while processing your request."))
        flag_note = f"<p style='color:orange;font-weight:bold;'>Error: {str(e)}</p>"

    end_time = time.time()
    final_time = end_time - start_time
    print("latency in ms", final_time)

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

# Mount Gradio under /gradio
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True, workers=1)
