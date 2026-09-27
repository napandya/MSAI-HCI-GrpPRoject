"""StreamHub: prompt-only support demo with bounded context and CPU controls."""
import logging
import os
import queue
import threading
import time

import gradio as gr
import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, StoppingCriteria,
    StoppingCriteriaList, TextIteratorStreamer,
)

from support import build_prompt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
MODEL_NAME = os.getenv("STREAMHUB_MODEL", "google/flan-t5-large")
MAX_NEW_TOKENS = int(os.getenv("STREAMHUB_MAX_NEW_TOKENS", "60"))
REQUEST_TIMEOUT = float(os.getenv("STREAMHUB_TIMEOUT", "90"))
if MAX_NEW_TOKENS <= 0 or REQUEST_TIMEOUT <= 0:
    raise ValueError("Token limit and timeout must be positive.")
if os.getenv("STREAMHUB_CPU_THREADS"):
    torch.set_num_threads(int(os.environ["STREAMHUB_CPU_THREADS"]))

model = tokenizer = None
generation_lock = threading.Lock()


def load_model():
    global model, tokenizer
    if model is None:
        started = time.perf_counter()
        loaded_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        loaded_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        loaded_model.eval()
        tokenizer, model = loaded_tokenizer, loaded_model
        log.info("Loaded %s in %.2fs; CPU threads=%s", MODEL_NAME,
                 time.perf_counter() - started, torch.get_num_threads())


class StopRequest(StoppingCriteria):
    def __init__(self, event, deadline):
        self.event, self.deadline = event, deadline

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set() or time.perf_counter() >= self.deadline


def respond(message, history):
    if not message or not message.strip():
        yield "What playback, subscription, or billing question can I help with?"
        return
    if model is None or tokenizer is None:
        yield "The support assistant could not load. Please try again after the app is restarted."
        return
    if not generation_lock.acquire(blocking=False):
        yield "The assistant is finishing another request. Please try again shortly."
        return

    worker = None
    cancelled = threading.Event()
    try:
        try:
            prompt = build_prompt(message.strip(), history,
                                  lambda text: len(tokenizer.encode(text)))
        except ValueError as exc:
            yield str(exc)
            return
        inputs = tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, timeout=0.25)
        errors = queue.Queue()
        done = threading.Event()
        started = time.perf_counter()
        deadline = started + REQUEST_TIMEOUT

        def generate():
            try:
                with torch.inference_mode():
                    model.generate(
                        **inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False, num_beams=1, use_cache=True,
                        stopping_criteria=StoppingCriteriaList([StopRequest(cancelled, deadline)]),
                    )
            except Exception as exc:
                log.exception("Generation failed")
                errors.put(exc)
            finally:
                done.set()
                generation_lock.release()

        worker = threading.Thread(target=generate, daemon=True)
        try:
            worker.start()
        except Exception:
            worker = None
            log.exception("Could not start generation")
            yield "I couldn't start that response. Please try again."
            return
        text = ""
        first_text = None
        while True:
            if not errors.empty():
                yield "I couldn't finish that response. Please try again."
                return
            if time.perf_counter() >= deadline:
                yield "This response took too long. Please try a shorter question or try again shortly."
                return
            try:
                chunk = next(streamer)
            except queue.Empty:
                if done.is_set():
                    if not errors.empty():
                        yield "I couldn't finish that response. Please try again."
                        return
                    break
                continue
            except StopIteration:
                break
            text += chunk
            if text.strip():
                if first_text is None:
                    first_text = time.perf_counter() - started
                yield text
        if not text.strip():
            yield "I couldn't produce an answer. Please rephrase your question."
        log.info("model=%s first_text_s=%s total_s=%.2f input_tokens=%s",
                 MODEL_NAME, first_text, time.perf_counter() - started,
                 inputs["input_ids"].shape[-1])
    finally:
        cancelled.set()
        # The worker owns the lock until generation actually stops, including
        # after UI cancellation or timeout. Never start overlapping CPU work.
        if worker is None:
            generation_lock.release()
        else:
            worker.join(timeout=0.1)


with gr.Blocks() as demo:
    gr.ChatInterface(
        fn=respond,
        title="StreamHub Support Bot",
        description=(
            "I'm StreamHub's virtual support assistant. I can guide playback "
            "troubleshooting and explain subscription and billing policies. "
            "StreamHub is a fictional class-project service. I can't access "
            "accounts, change plans, or take payments. Don't share passwords "
            "or payment details."
        ),
        examples=["The app keeps freezing. What should I try?",
                  "When will my plan change take effect?",
                  "How long is the free trial?"],
        cache_examples=False,
        concurrency_limit=1,
        textbox=gr.Textbox(label="Your support question", placeholder="Ask a support question…",
                           max_length=2000),
    )

demo.queue(max_size=8)

if __name__ == "__main__":
    try:
        load_model()
    except Exception:
        log.exception("Model loading failed")
    demo.launch(theme=gr.themes.Soft(primary_hue="blue"))
