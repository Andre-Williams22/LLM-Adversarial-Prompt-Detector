"""Gradio demo interface.

Presentation only. The interface is a thin shell over utils.chat_handler so the
detection path stays testable and reusable behind the JSON API.
"""
import gradio as gr

from utils.chat_handler import chat_and_detect

STYLES = """
<style>
    #chatbox { background-color: #f0f0f0; border-radius: 5px; padding: 10px; }
    #chatbox .bot { background-color: #e0e0e0; border-radius: 5px; padding: 5px; margin: 5px 0; }
    #chatbox .user { background-color: #fff; border-radius: 5px; padding: 5px; margin: 5px 0; }
    #user_input { background: #f0f0f0; border: 1px solid #ccc; border-radius: 5px; }
    #send_button { background: #007BFF; color: #fff; border: none; border-radius: 5px;
                   padding: 10px 20px; cursor: pointer; }
    #send_button:hover { background: #0056b3; }
</style>
"""

HEADER = """
<h1 style="text-align:center;color:#007BFF;">Adversarial Prompt Detector</h1>
<p style="text-align:center;color:#555;">
    Every prompt is screened by a four-signal ensemble before it reaches the
    assistant. The verdict panel shows which models fired and why.
</p>
"""


def build_interface(handler=chat_and_detect) -> gr.Blocks:
    """Construct the Gradio Blocks app. The handler is injectable for testing."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Adversarial Prompt Detector") as demo:
        gr.HTML(STYLES)
        gr.Markdown(HEADER)

        # The handler yields (speaker, message) pairs, so the tuples format is
        # requested explicitly rather than left to Gradio's shifting default.
        chatbot = gr.Chatbot(
            label="Screened conversation", elem_id="chatbox", type="tuples"
        )
        state = gr.State([])
        user_input = gr.Textbox(
            placeholder="Type a prompt to screen it...",
            label="Your message",
            lines=2,
            max_lines=5,
            elem_id="user_input",
        )
        send_button = gr.Button("Send", elem_id="send_button")
        verdict = gr.Markdown("", elem_id="verdict")

        send_button.click(
            fn=handler,
            inputs=[user_input, state],
            outputs=[chatbot, state, verdict],
            queue=False,
        )

    return demo
