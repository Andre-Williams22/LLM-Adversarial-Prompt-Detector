import os
from dotenv import load_dotenv

import openai 
from openai import OpenAI
import mlflow
import torch
import gradio as gr
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Load environment variables
load_dotenv()
MODEL_DIR = os.getenv("MODEL_PATH", "outputs/electra/best_model")
BASE_MODEL = "google/electra-small-discriminator"

# Set up MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
mlflow.set_experiment("adversarial_prompt_detector")

# Load detector model and tokenizer
base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
detector = PeftModel.from_pretrained(base, MODEL_DIR)
detector.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Initialize OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # OpenAI Python client

def chat_and_detect(user_message, history):
    # history = history or []
    # history.append(("User", user_message))

    try:
        # Detection logic using the pre-trained detector model
        inputs = tokenizer(user_message, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = detector(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            print("scores", scores)
            print("scores", scores[0][1])
            is_adversarial = scores[0][1] > 0.5  # Assuming index 1 is adversarial
            
        if is_adversarial:
            flag_note = (
                '<p style="color: red; font-weight: bold; font-size: 18px;">'
                "⚠️ Adversarial prompt detected! The request was not processed. Please remove harmful speech from the prompt. And try again!"
                "</p>"
            )
            history.append(("Bot", f"Adversarial prompt detected! I'm sorry, but I can't help with that."))
            return history, history, flag_note
        else:
            flag_note = (
                '<p style="color: green; font-weight: bold; font-size: 18px;">'
                "✅ No adversarial prompt detected. Processing request..."
                "</p>"
            )
            history.append(("Bot", "No adversarial prompt detected. Processing your request..."))
            # # OpenAI API call using the correct method
            # response = client.chat.completions.create(model="gpt-4.1",
            # messages=[{"role": "system", "content": "You are helpful AI assistant called Chat-GPT and are there to answer every question to the best of your ability."}]
            #         + [{"role": "user", "content": msg} for _, msg in history])
            # bot_reply = response.choices[0].message.content
            # history.append(("Bot", bot_reply))

            # # Placeholder for detector logic, MLflow logging, and flagging
            # note = "Detection logic not implemented yet."

            note = "Detection logic not implemented yet."
            history = [('User', f"{user_message}"), ('Bot', f'{user_message}')]
            # print(f"Chat History: {history}, note {note}")

    except Exception as e:
        bot_reply = "An error occurred while processing your request."
        history.append(("Bot", bot_reply))

        flag_note = (
            '<p style="color: orange; font-weight: bold; font-size: 18px;">'
            f"Error: {str(e)}"
            "</p>"
        )

    return history, history, note

# Build FastAPI + Gradio app
app = FastAPI()


@app.get("/health", summary="Health check", tags=["Health"])
async def health_check():
    """
    Basic liveness check. Returns OK if the app is running
    and the model is loaded.
    """
    try:
        # Try a dummy inference to ensure your detector is responsive
        _ = tokenizer("ping", return_tensors="pt")
        with torch.no_grad():
            _ = detector(**_).logits
        return {"status": "ok"}
    except Exception as e:
        # If something’s wrong (e.g., model not loaded), surface a 503
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.get("/")
def home():
    return {"message": "Welcome to the ChatGPT + Adversarial Prompt Detector API!"}

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <style>
            #chatbox {
                background-color: #f0f0f0; /* Light grey background for chatbox */
                border-radius: 5px;
                padding: 10px;
            }

            #chatbox .bot {
                background-color: #e0e0e0; /* Slightly darker grey for bot messages */
                border-radius: 5px;
                padding: 5px;
                margin: 5px 0;
}

            #chatbox .user {
                background-color: #ffffff; /* White background for user messages */
                border-radius: 5px;
                padding: 5px;
                margin: 5px 0;
            }

            #user_input {
                background-color: #f0f0f0; /* Light grey background for user input */
                border: 1px solid #ccc;
                border-radius: 5px;
            }

            #send_button {
                background-color: #007BFF; /* Blue background for button */
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                cursor: pointer;
            }

            #send_button:hover {
                background-color: #0056b3; /* Darker blue on hover */
            }
        </style>
        """
    )

    gr.Markdown(
        """
        <h1 style="text-align: center; color: #007BFF;">ChatGPT + Adversarial Prompt Detector</h1>
        <p style="text-align: center; color: #555;">An AI assistant integrated with a detector for adversarial prompts.</p>
        """
    )
    chatbot = gr.Chatbot(label="ChatGPT + Detector", elem_id="chatbox")
    state = gr.State([])  # Holds chat history
    user_in = gr.Textbox(
        placeholder="Ask anything here ....",
        label="Your Message",
        lines=2,
        max_lines=5,
        elem_id="user_input",
    )
    send_btn = gr.Button("Send", elem_id="send_button")
    flag_note = gr.Markdown(
        "", elem_id="flag_note"
    )  # Markdown for flagging notes

    send_btn.click(
        fn=chat_and_detect,
        inputs=[user_in, state],
        outputs=[chatbot, state, flag_note],
        queue=False,
    )

# Mount the Gradio app under /gradio
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)