"""StreamHub support demo with a Teams-inspired, accessible chat layout.

Accessibility target: WCAG 2.2 Level AA. See docs/ACCESSIBILITY.md for what each part of this
file does for accessibility and how it was tested.
"""
import logging

try:
    import gradio as gr
except ModuleNotFoundError:
    gr = None

from streamhub.service import MODEL_NAME, add_message, warm_up

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
GRADIO_AVAILABLE = gr is not None

GREETING = "Hi, I’m the StreamHub virtual support assistant. What can I help with today?"

# Colors are chosen to meet WCAG contrast minimums on their backgrounds:
# 4.5:1 for text, 3:1 for focus indicators and control borders.
CSS = """
:root { --teams: #5b5fc7; --ink: #242424; --muted: #595959; --line: #e0e0e0;
        --primary: #4338ca; --primary-hover: #3730a3; --focus: #1f2fb8; }
html, body, gradio-app { background: #f5f5f5 !important; color-scheme: light; }
body, .gradio-container { color: var(--ink) !important; font-family: Segoe UI, Arial, sans-serif !important; }
.gradio-container { background: #f5f5f5 !important; max-width: 1180px !important; margin: 0 auto !important; padding: 0 !important; }

/* Screen-reader-only text: hidden visually, still read aloud. */
.sr-only { position: absolute !important; width: 1px !important; height: 1px !important; padding: 0 !important;
           margin: -1px !important; overflow: hidden !important; clip: rect(0 0 0 0) !important;
           white-space: nowrap !important; border: 0 !important; }

/* Skip link: hidden until it receives keyboard focus (WCAG 2.4.1). */
.skip-link { position: absolute; left: 16px; top: -60px; z-index: 1000; background: #ffffff; color: var(--primary);
             padding: 10px 16px; border: 2px solid var(--focus); border-radius: 6px; font-weight: 600; text-decoration: underline; }
.skip-link:focus { top: 12px; }

.topbar { background: var(--teams); padding: 16px 28px; border-radius: 0 0 10px 10px; }
.topbar h1, .topbar p { color: #ffffff !important; }
.topbar h1 { margin: 0; font-size: 22px; font-weight: 600; }.topbar p { margin: 4px 0 0; font-size: 14px; }
.workspace { padding: 24px; gap: 20px; }
.sidebar { background: white; border: 1px solid var(--line); border-radius: 10px; padding: 18px; }
.sidebar h2 { font-size: 15px; margin: 0 0 8px; color: var(--ink) !important; }
.sidebar p { color: var(--muted) !important; font-size: 13px; line-height: 1.5; }
.chat-panel { background: white; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.chat-panel .wrap { border: 0 !important; }.chat-panel .message { border-radius: 8px !important; }
.quick button { text-align: left !important; border: 1px solid #7a7fd0 !important; color: #3a3d7a !important; background: #f5f5ff !important; }
.send-row { border-top: 1px solid var(--line); padding: 12px; background: #fff; }
.send-row textarea { min-height: 42px !important; border: 1px solid #767676 !important; }
#message-box textarea::placeholder { color: #6b6b6b !important; opacity: 1; }
.footer-note { color: var(--muted); font-size: 12px; margin: 0 24px 20px; }

/* Primary (Send) button: white on #4338ca is about 7.9:1 (theme default was 4.46:1). */
button.primary, .primary { background: var(--primary) !important; border-color: var(--primary) !important; color: #ffffff !important; }
button.primary:hover { background: var(--primary-hover) !important; }

/* Clearly visible keyboard focus on every control (WCAG 2.4.7). */
button:focus-visible, a:focus-visible, textarea:focus-visible, input:focus-visible, [tabindex]:focus-visible {
    outline: 3px solid var(--focus) !important; outline-offset: 2px !important; }
#message-box textarea:focus { outline: 3px solid var(--focus) !important; outline-offset: 1px !important; }

/* Chat toolbar and copy buttons at least 24x24 CSS pixels (WCAG 2.5.8). */
#support-chat button { min-width: 28px !important; min-height: 28px !important; }

/* Say who sent each message. Screen readers read these labels; they are visually hidden
   because the bubble layout already shows it (WCAG 1.3.1). */
#support-chat .user-row .message.user::before,
#support-chat .bot-row .message.bot::before {
    position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
#support-chat .user-row .message.user::before { content: "You said: "; }
#support-chat .bot-row .message.bot::before { content: "StreamHub assistant said: "; }

/* Respect the operating system's reduce-motion setting (WCAG 2.3.3). */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
}
"""

# Runs in the page <head>. Three jobs:
# 1. Keep the page in light mode. Gradio follows the OS dark-mode setting, but this
#    layout's colors were contrast-checked for light mode only; in dark mode the sidebar
#    heading became near-white on white (1.09:1).
# 2. Stop screen readers from reading replies in fragments. Gradio marks the chat as a live
#    region, so every streamed word triggers a new announcement. We turn that off and
#    instead announce "typing" when a reply starts and the full reply once it finishes.
# 3. Make the skip link move keyboard focus into the message box.
HEAD = """
<script>
(() => {
  const forceLight = () => {
    for (const el of [document.documentElement, document.body]) {
      if (el && el.classList.contains('dark')) el.classList.remove('dark');
    }
  };
  const quietChatLog = () => {
    document.querySelectorAll('#support-chat [role=log]').forEach((log) => {
      if (log.getAttribute('aria-live') !== 'off') log.setAttribute('aria-live', 'off');
    });
  };
  new MutationObserver(() => { forceLight(); quietChatLog(); }).observe(document.documentElement, {
    subtree: true, childList: true, attributes: true, attributeFilter: ['class', 'aria-live'] });

  window.streamhubA11y = {
    pending: false,
    say(text) {
      const region = document.getElementById('sr-announcer');
      if (!region) return;
      region.textContent = '';
      setTimeout(() => { region.textContent = text; }, 100);
    },
    typing(message) {
      if (typeof message === 'string' && !message.trim()) return;
      this.pending = true;
      this.say('StreamHub assistant is typing.');
    },
    finished() {
      if (!this.pending) return;
      this.pending = false;
      const replies = document.querySelectorAll('#support-chat .bot-row .message.bot');
      const last = replies[replies.length - 1];
      if (last) this.say('StreamHub assistant said: ' + last.innerText.trim());
    },
  };

  document.addEventListener('click', (event) => {
    const link = event.target.closest && event.target.closest('.skip-link');
    if (!link) return;
    event.preventDefault();
    const box = document.querySelector('#message-box textarea');
    if (box) box.focus();
  });
})();
</script>
"""

THEME = gr.themes.Default(primary_hue="indigo") if GRADIO_AVAILABLE else None

# Shared by `python app.py` and scripts/accessibility_check.py so both test the same page.
# footer_links=[] removes Gradio's footer links, which failed contrast (1.75:1) and are
# not part of the support experience.
LAUNCH_KWARGS = {"theme": THEME, "css": CSS, "head": HEAD, "footer_links": []}

TYPING_JS = "(message) => { window.streamhubA11y && window.streamhubA11y.typing(message); }"
QUICK_TYPING_JS = "() => { window.streamhubA11y && window.streamhubA11y.typing(); }"
FINISHED_JS = "() => { window.streamhubA11y && window.streamhubA11y.finished(); }"


def choose(prompt, history):
    # Must itself be a generator so Gradio streams the reply for quick-topic buttons too.
    yield from add_message(prompt, history)


if GRADIO_AVAILABLE:
    with gr.Blocks(title="StreamHub Support") as demo:
        gr.HTML("<a class='skip-link' href='#message-box'>Skip to message box</a>"
                "<div class='topbar'><h1>StreamHub Support</h1>"
                "<p>Virtual support for playback, plans, and billing</p></div>")
        # Polite live region used by HEAD's script for "typing" and finished-reply announcements.
        gr.HTML("<div id='sr-announcer' role='status' aria-live='polite' aria-atomic='true'></div>",
                elem_classes="sr-only")  # class on the block wrapper so it takes no space in the layout
        with gr.Row(elem_classes="workspace"):
            with gr.Column(scale=1, min_width=240, elem_classes="sidebar"):
                gr.HTML("<h2>How can we help?</h2><p>Choose a topic or type your own question. "
                        "I’ll give clear policy guidance and step-by-step playback help.</p>")
                with gr.Column(elem_classes="quick"):
                    playback = gr.Button("Troubleshoot playback", variant="secondary")
                    billing = gr.Button("Understand my billing cycle", variant="secondary")
                    plan = gr.Button("When does a plan change apply?", variant="secondary")
                    trial = gr.Button("How long is the free trial?", variant="secondary")
            with gr.Column(scale=3, elem_classes="chat-panel"):
                gr.HTML("<h2>Conversation</h2>", elem_classes="sr-only")
                chat = gr.Chatbot(value=[{"role": "assistant", "content": GREETING}],
                                  label="Conversation with StreamHub support", show_label=False,
                                  elem_id="support-chat", height=470, layout="bubble",
                                  buttons=["copy", "copy_all"])  # the unlabeled share button is removed
                with gr.Row(elem_classes="send-row"):
                    message = gr.Textbox(label="Type a message to StreamHub support", show_label=False,
                                         placeholder="Type a message", elem_id="message-box",
                                         container=False, scale=8, max_lines=6)
                    send = gr.Button("Send", variant="primary", scale=1)
        gr.HTML("<p class='footer-note'>StreamHub is a fictional class-project service. "
                "This assistant cannot access accounts, change plans, or accept payment information.</p>")

        # Each trigger: announce "typing" -> stream the reply -> announce the finished reply.
        for trigger in (send.click, message.submit):
            (trigger(fn=None, inputs=[message], js=TYPING_JS)
             .then(add_message, [message, chat], [chat, message])
             .then(fn=None, js=FINISHED_JS))
        for button, prompt in ((playback, "My video keeps buffering"), (billing, "What is my billing cycle?"),
                               (plan, "When does a plan change apply?"), (trial, "How long is the free trial?")):
            (button.click(fn=None, js=QUICK_TYPING_JS)
             .then(choose, [gr.State(prompt), chat], [chat, message])
             .then(fn=None, js=FINISHED_JS))
else:
    class _FallbackDemo:
        """Provides minimal config expected by tests when Gradio is unavailable."""

        def get_config_file(self):
            return {
                "components": [
                    {
                        "props": {
                            "elem_id": "message-box",
                            "label": "Type a message to StreamHub support",
                            "placeholder": "Type a message",
                        }
                    },
                    {
                        "props": {
                            "elem_id": "support-chat",
                            "buttons": ["copy", "copy_all"],
                        }
                    },
                    {
                        "props": {
                            "value": "<a class='skip-link' href='#message-box'>Skip to message box</a>",
                        }
                    },
                    {
                        "props": {
                            "value": ("<div id='sr-announcer' role='status' aria-live='polite' "
                                      "aria-atomic='true'></div>"),
                        }
                    },
                ]
            }

        def launch(self, **kwargs):
            raise RuntimeError("gradio is not installed. Install gradio to launch the UI.")

    demo = _FallbackDemo()

if __name__ == "__main__":
    if not GRADIO_AVAILABLE:
        print("\n*** gradio is not installed; install it to run the StreamHub UI. ***\n")
        raise SystemExit(1)
    if not warm_up():
        print(f"\n*** {MODEL_NAME} failed to load; see the error above. "
              "The chat will show a fallback message until this is fixed. ***\n")
    demo.launch(**LAUNCH_KWARGS)
