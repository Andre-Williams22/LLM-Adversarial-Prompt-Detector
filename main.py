from fastapi import FastAPI
import gradio as gr

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the ChatGPT + Adversarial Prompt Detector API!"}

def chat_and_detect(user_input, history):
    # Example function to simulate chat and detection
    history = history or []
    bot_response = f"Echo: {user_input}"  # Replace with actual bot logic
    history.append(("User", user_input))
    history.append(("Bot", bot_response))
    flag_note = "No adversarial prompt detected."  # Replace with actual detection logic
    return history, history, flag_note

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
