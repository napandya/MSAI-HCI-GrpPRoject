"""StreamHub support demo with a Teams-inspired, accessible chat layout."""
import gradio as gr

from support import add_message

CSS = """
:root { --teams: #5b5fc7; --ink: #242424; --muted: #616161; --line: #e0e0e0; }
body, .gradio-container { background: #f5f5f5 !important; color: var(--ink) !important; font-family: Segoe UI, Arial, sans-serif !important; }
.gradio-container { max-width: 1180px !important; padding: 0 !important; }
.topbar { background: var(--teams); color: white; padding: 16px 28px; }.topbar h1 { margin: 0; font-size: 22px; font-weight: 600; }.topbar p { margin: 4px 0 0; font-size: 14px; opacity: .92; }
.workspace { padding: 24px; gap: 20px; }.sidebar { background: white; border: 1px solid var(--line); border-radius: 10px; padding: 18px; }.sidebar h2 { font-size: 15px; margin: 0 0 8px; }.sidebar p { color: var(--muted); font-size: 13px; line-height: 1.45; }
.chat-panel { background: white; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }.chat-panel .wrap { border: 0 !important; }.chat-panel .message { border-radius: 8px !important; }
.quick button { text-align: left !important; border-color: #d1d1f0 !important; color: #424584 !important; background: #f5f5ff !important; }.send-row { border-top: 1px solid var(--line); padding: 12px; background: #fff; }.send-row textarea { min-height: 42px !important; }.footer-note { color: #616161; font-size: 12px; margin: 0 24px 20px; }
"""


def choose(prompt, history):
    return add_message(prompt, history)


with gr.Blocks(title="StreamHub Support") as demo:
    gr.HTML("<div class='topbar'><h1>StreamHub Support</h1><p>Virtual support for playback, plans, and billing</p></div>")
    with gr.Row(elem_classes="workspace"):
        with gr.Column(scale=1, min_width=240, elem_classes="sidebar"):
            gr.HTML("<h2>How can we help?</h2><p>Choose a topic or type your own question. I’ll give clear policy guidance and step-by-step playback help.</p>")
            with gr.Column(elem_classes="quick"):
                playback = gr.Button("Troubleshoot playback", variant="secondary")
                billing = gr.Button("Understand my billing cycle", variant="secondary")
                plan = gr.Button("When does a plan change apply?", variant="secondary")
                trial = gr.Button("How long is the free trial?", variant="secondary")
        with gr.Column(scale=3, elem_classes="chat-panel"):
            chat = gr.Chatbot(value=[{"role": "assistant", "content": "Hi, I’m the StreamHub virtual support assistant. What can I help with today?"}], height=470, show_label=False, layout="bubble")
            with gr.Row(elem_classes="send-row"):
                message = gr.Textbox(placeholder="Type a message", show_label=False, container=False, scale=8)
                send = gr.Button("Send", variant="primary", scale=1)
    gr.HTML("<p class='footer-note'>StreamHub is a fictional class-project service. This assistant cannot access accounts, change plans, or accept payment information.</p>")

    send.click(add_message, [message, chat], [chat, message])
    message.submit(add_message, [message, chat], [chat, message])
    for button, prompt in ((playback, "My video keeps buffering"), (billing, "What is my billing cycle?"), (plan, "When does a plan change apply?"), (trial, "How long is the free trial?")):
        button.click(choose, [gr.State(prompt), chat], [chat, message])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(primary_hue="indigo"), css=CSS)
