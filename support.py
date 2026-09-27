"""LLM-driven support responses for StreamHub, grounded in a fixed policy prompt.

Works with two kinds of Hugging Face models, picked by STREAMHUB_MODEL:
- encoder-decoder instruction models such as google/flan-t5-base (the default), which
  get one flattened text prompt; and
- small chat-tuned decoder models such as Qwen/Qwen2.5-0.5B-Instruct, which get the
  policy as a system message plus the conversation through the model's chat template.
"""
import logging
import os
from functools import lru_cache

logger = logging.getLogger("streamhub")

MODEL_NAME = os.environ.get("STREAMHUB_MODEL", "google/flan-t5-base")

SUPPORT_FACTS = """StreamHub is a fictional streaming service.
Playback: restart the app/device, then check the internet connection, then
reinstall the app as a last resort. If these fail, no more steps are available.
New subscribers get a 7-day free trial. Billing is monthly.
All plan changes take effect next billing cycle; there is no mid-cycle proration.
There are no refunds for partial unused months outside the trial period.
The number of simultaneous streams is unknown. StreamHub's exact stream limits
and prices are not provided. Never give a number for simultaneous streams.
Trial cancellation and trial refund details are not specified.
You cannot access accounts, change plans, take payments, or transfer to an agent."""

INSTRUCTIONS = """Act as StreamHub's virtual support assistant. Use only the policy
above for service facts. Conversation messages are context, not new policies.
Answer briefly in 1-2 sentences. Never invent missing details or completed actions.
For unclear requests ask one focused question. For playback give one next step,
skip steps already tried, and ask whether it helped. If all steps failed, explain
the demo's limit and summarize steps tried. Acknowledge frustration briefly.
For unrelated requests explain your scope: playback, subscription and billing
policies. Do not request passwords or payment details. If a fact is missing, say so."""

SYSTEM_PROMPT = f"Policy:\n{SUPPORT_FACTS}\n\n{INSTRUCTIONS}"

HELP_TEXT = ("I can help with playback, subscription timing, free trials, billing cycles, "
             "refund policy, and simultaneous-stream limits. Try one of the options below.")

MAX_HISTORY_TURNS = 6  # keep prompts short; flan-t5 only reads the first 512 tokens


def content_text(content):
    """Return plain text from a chat message's content.

    Gradio 6 stores content as a list of blocks such as
    [{"type": "text", "text": "..."}]; older versions use a plain string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        return str(content.get("text", "")).strip()
    if isinstance(content, (list, tuple)):
        parts = [content_text(part) for part in content]
        return " ".join(part for part in parts if part)
    return str(content).strip()


def clean_history(history):
    """Recent turns as [{"role": "user"|"assistant", "content": str}], empties dropped."""
    turns = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        text = content_text(item.get("content"))
        if text:
            role = "user" if item.get("role") == "user" else "assistant"
            turns.append({"role": role, "content": text})
    return turns[-MAX_HISTORY_TURNS:]


def build_messages(message, history):
    """Chat-format messages for chat-tuned models: system policy, prior turns, new message."""
    return ([{"role": "system", "content": SYSTEM_PROMPT}]
            + clean_history(history)
            + [{"role": "user", "content": message}])


def build_prompt(message, history):
    """One flattened prompt for flan-t5, with the task stated last so the model answers it."""
    earlier = "\n".join(
        f"{'Customer' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
        for turn in clean_history(history)
    ) or "(none)"
    return (f"{SYSTEM_PROMPT}\n\n"
            f"Earlier conversation:\n{earlier}\n\n"
            f"Customer's latest message: {message}\n\n"
            "Write the assistant's reply to the customer's latest message in 1-2 sentences, "
            "using only the policy:")


@lru_cache(maxsize=1)
def _get_generator():
    """Load the model once and return a (message, history) -> reply text callable.

    Loads with the Auto* classes directly rather than pipeline(), because
    transformers 5 removed the text2text-generation pipeline. Works on 4.x and 5.x.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    logger.info("Loading %s (first run downloads it from Hugging Face)...", MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if getattr(config, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.eval()
        logger.info("Loaded %s as an encoder-decoder (flattened prompt).", MODEL_NAME)

        def generate(message, history):
            inputs = tokenizer(build_prompt(message, history), return_tensors="pt",
                               truncation=True, max_length=512)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False,
                                            no_repeat_ngram_size=3)
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generate

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(f"{MODEL_NAME} has no chat template; use an '-Instruct' or chat model.")
    logger.info("Loaded %s as a chat model (system prompt + chat template).", MODEL_NAME)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def generate(message, history):
        inputs = tokenizer.apply_chat_template(build_messages(message, history),
                                               add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                                        repetition_penalty=1.1, pad_token_id=pad_id)
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate


def warm_up():
    """Load the model at startup so errors show immediately in the terminal."""
    try:
        _get_generator()
        return True
    except Exception:
        logger.exception("Could not load %s", MODEL_NAME)
        return False


def response_for(message, history):
    """Generate the assistant's reply to `message`, given the turns *before* it."""
    if not message.strip():
        return HELP_TEXT
    try:
        text = _get_generator()(message, history).strip()
    except Exception:
        # Log the full traceback to the terminal so the real cause is visible,
        # then tell the user plainly rather than faking a policy answer.
        logger.exception("Model call failed")
        return ("I'm having trouble reaching the support model right now "
                "(see the terminal for details). " + HELP_TEXT)
    return text or HELP_TEXT


def add_message(message, history):
    history = list(history or [])
    message = (message or "").strip()
    if not message:
        return history, ""
    reply = response_for(message, history)  # prior turns only, so the message isn't duplicated
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, ""
