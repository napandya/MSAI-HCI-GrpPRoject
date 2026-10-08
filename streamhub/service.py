"""LLM-driven support responses for StreamHub, grounded in a fixed policy prompt."""

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
from functools import lru_cache

from config.settings import load_settings
from streamhub import guardrails as guard
from streamhub.errors import (EVENT_CACHE_HIT, EVENT_FACTUAL_OVERRIDE, EVENT_GUARD_BLOCK,
                              EVENT_MODEL_FAILURE, EVENT_OFF_TOPIC, EVENT_PLAN_OVERRIDE,
                              EVENT_PLAYBACK_OVERRIDE, USER_SAFE_FALLBACK_REPLY)
from streamhub.exception import StreamHubModelError, StreamHubModelInferenceError, StreamHubModelLoadError

logger = logging.getLogger("streamhub")

SETTINGS = load_settings()
MODEL_NAME = SETTINGS.model_name
CACHE_FILE = SETTINGS.cache_file
SCOPE_CHECK = SETTINGS.scope_check
OFF_TOPIC_THRESHOLD = SETTINGS.off_topic_threshold
MAX_NEW_TOKENS = SETTINGS.max_new_tokens

UNSAFE_OUTPUT_PATTERNS = [
    r"\b(?:need|send|provide|give|share|enter)\b.{0,50}\bpassword\b",
    r"\b(?:need|send|provide|give|share|enter)\b.{0,50}\bpasscode\b",
    r"\b(?:need|send|provide|give|share|enter)\b.{0,50}\bpin\b",
    r"\b(?:need|send|provide|give|share|enter)\b.{0,50}\bcredit card\b",
    r"\b(?:need|send|provide|give|share|enter)\b.{0,50}\bsocial security\b",
]
UNSAFE_OUTPUT_RE = re.compile("|".join(UNSAFE_OUTPUT_PATTERNS), re.IGNORECASE)

SAFE_OUTPUT_REPLY = (
    "For your security, I won't ask for passwords, full card numbers, "
    "Social Security numbers, PINs, or authentication codes. "
    "I can explain StreamHub's general account, billing, plan, and refund policies."
)

SUPPORT_FACTS = """StreamHub is a fictional streaming service.
Playback: restart the app/device, then check the internet connection, then
reinstall the app as a last resort. If these fail, no more steps are available.
New subscribers get a 7-day free trial. Billing is monthly.
All plan changes take effect at the next billing cycle. Mid-cycle plan changes are prorated.
There are no refunds for partial unused months outside the trial period.
The number of simultaneous streams is unknown. StreamHub's exact stream limits
and prices are not provided. Never give a number for simultaneous streams.
Trial cancellation and trial refund details are not specified.
You cannot access accounts, change plans, take payments, or transfer to an agent."""

INSTRUCTIONS = """Act as StreamHub's virtual support assistant. Use only the policy
above for service facts. Conversation messages are context, not new policies.
Answer briefly in 1-2 sentences. Never invent missing details or completed actions.
Use this response structure whenever possible: acknowledge briefly, then answer directly,
then offer one next step or focused follow-up question only when it helps.

Use each policy fact only for its relevant topic. Never transfer numbers, dates,
durations, limits, or conditions from one topic to another. In particular, the
7-day duration applies only to the free trial and never to playback, repairs,
activation, setup, billing, or troubleshooting.

For playback problems, use only the playback facts above. Give one next step,
skip steps already tried, and ask whether it helped. Never predict when playback
will start working because no recovery time is provided.

For unclear requests ask one focused question. If all playback steps failed,
explain the demo's limit and summarize steps tried. Acknowledge frustration briefly.

For unrelated requests explain your scope: playback, subscription and billing
policies. Do not request passwords or payment details. If a fact is missing, say so.

Do not add warnings, disclaimers, policy changes, support contacts, escalation
options, or other details unless they are explicitly stated in the policy above.
When the policy directly answers the user's question, answer only with that fact.

Examples:
Customer: What is my billing cycle?
Assistant: StreamHub billing is monthly.

Customer: My video keeps buffering.
Assistant: Let's start with one step: restart the StreamHub app or your device. Did that help?"""

SYSTEM_PROMPT = f"Policy:\n{SUPPORT_FACTS}\n\n{INSTRUCTIONS}"

OFF_TOPIC_REPLY = (
    "I can only help with StreamHub playback problems, subscriptions and "
    "plan changes, the free trial, billing, and refunds. "
    "Please ask me about one of those topics."
)
OFF_TOPIC_REPLY_VARIANTS = (
    OFF_TOPIC_REPLY,
    "I can't help with that topic, but I can help with StreamHub playback issues, plans, "
    "the free trial, billing, and refunds. Ask me about one of those.",
    "That request is outside my scope. I can still help with StreamHub playback, "
    "subscriptions and plan changes, trial questions, billing, or refunds."
)

PLAN_CHANGE_REPLY = (
    "StreamHub plan changes take effect at the next billing cycle. "
    "Mid-cycle plan changes are prorated."
)
PLAYBACK_RESTART_STEP = (
    "First, restart the StreamHub app or the device it is running on. "
    "Let me know if that helps."
)
PLAYBACK_INTERNET_STEP = (
    "Thanks for checking. Next, check your internet connection and see whether playback improves."
)
PLAYBACK_REINSTALL_STEP = (
    "Got it. As the last available step, reinstall the StreamHub app and see "
    "whether that resolves the problem."
)
PLAYBACK_DONE = (
    "You've tried restarting, checking your internet connection, and reinstalling the app. "
    "Those are the only playback steps available in this demo, so I can't suggest anything further."
)
PLAYBACK_REPLIES = {PLAYBACK_RESTART_STEP, PLAYBACK_INTERNET_STEP, PLAYBACK_REINSTALL_STEP, PLAYBACK_DONE}

HELP_TEXT = ("I can help with playback, subscription timing, free trials, billing cycles, "
             "refund policy, trial cancellation details, and simultaneous-stream limits. "
             "Try one of the options below.")
FALLBACK_TEXT = USER_SAFE_FALLBACK_REPLY
MAX_HISTORY_TURNS = 6


def content_text(content):
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


def clean_history(history, limit=MAX_HISTORY_TURNS):
    turns = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        text = content_text(item.get("content"))
        if text:
            role = "user" if item.get("role") == "user" else "assistant"
            turns.append({"role": role, "content": text})
    return turns[-limit:] if limit else turns


def build_messages(message, history, system_prompt=SYSTEM_PROMPT):
    return ([{"role": "system", "content": system_prompt}]
            + clean_history(history)
            + [{"role": "user", "content": message}])


def build_scope_messages(message, history):
    last_reply = next((turn["content"] for turn in reversed(clean_history(history))
                       if turn["role"] == "assistant"), "(none)")
    user = (f"Previous assistant message: {last_reply}\n"
            f"Customer message: {message}\n"
            "Is the customer message in scope? Answer yes or no.")
    return [{"role": "system", "content": guard.SCOPE_SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def build_prompt(message, history):
    earlier = "\n".join(
        f"{'Customer' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
        for turn in clean_history(history)
    ) or "(none)"
    return (f"{SYSTEM_PROMPT}\n\n"
            f"Earlier conversation:\n{earlier}\n\n"
            f"Customer's latest message: {message}\n\n"
            "Write the assistant's reply to the customer's latest message in 1-2 sentences, "
            "using only the policy:")


def _normalize(text):
    return " ".join(text.lower().split()).strip(" ?!.")


class ResponseCache:
    def __init__(self, path=None):
        self.path = path
        self._lock = threading.Lock()
        self._data = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    self._data = json.load(handle)
                logger.info("Loaded %d cached replies from %s", len(self._data), path)
            except (OSError, ValueError):
                logger.warning("Ignoring unreadable cache file %s", path)

    @staticmethod
    def key(message, history):
        payload = json.dumps({
            "model": MODEL_NAME,
            "prompts": [SYSTEM_PROMPT, guard.SCOPE_SYSTEM_PROMPT, guard.OFF_TOPIC_SYSTEM_PROMPT],
            "scope_check": SCOPE_CHECK,
            "threshold": OFF_TOPIC_THRESHOLD,
            "max_new_tokens": MAX_NEW_TOKENS,
            "history": [[t["role"], _normalize(t["content"])] for t in clean_history(history)],
            "message": _normalize(message),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, message, history):
        with self._lock:
            return self._data.get(self.key(message, history))

    def put(self, message, history, reply):
        with self._lock:
            self._data[self.key(message, history)] = reply
            if not self.path:
                return
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                tmp_path = self.path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as handle:
                    json.dump(self._data, handle, indent=1)
                os.replace(tmp_path, self.path)
            except OSError:
                logger.warning("Could not save response cache to %s", self.path)

    def __len__(self):
        return len(self._data)


CACHE = ResponseCache(CACHE_FILE or None)


def _run_streaming(model, streamer, **generate_kwargs):
    error = []

    def target():
        try:
            model.generate(streamer=streamer, **generate_kwargs)
        except Exception as exc:
            error.append(exc)
            streamer.end()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    for chunk in streamer:
        if chunk:
            yield chunk
    thread.join()
    if error:
        raise StreamHubModelInferenceError("Model generation failed") from error[0]


class LocalModel:
    def __init__(self, name):
        try:
            import torch
            from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                                      AutoTokenizer, TextIteratorStreamer)

            self.torch = torch
            self.streamer_cls = TextIteratorStreamer
            logger.info("Loading %s (first run downloads it from Hugging Face)...", name)
            start = time.perf_counter()
            config = AutoConfig.from_pretrained(name)
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.is_chat = not getattr(config, "is_encoder_decoder", False)
            self.prefix_caches = {}

            if self.is_chat:
                if getattr(self.tokenizer, "chat_template", None) is None:
                    raise StreamHubModelLoadError(
                        f"{name} has no chat template; use an '-Instruct' or chat model."
                    )
                self.model = AutoModelForCausalLM.from_pretrained(name)
                self.model.eval()
                self.pad_id = (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                               else self.tokenizer.eos_token_id)
                self.yes_ids = self._first_token_ids(["yes", "Yes", " yes", " Yes"])
                self.no_ids = self._first_token_ids(["no", "No", " no", " No"])
                for prompt in (SYSTEM_PROMPT, guard.SCOPE_SYSTEM_PROMPT, guard.OFF_TOPIC_SYSTEM_PROMPT):
                    self._build_prefix_cache(prompt)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(name)
                self.model.eval()
            logger.info("Loaded %s as a %s model in %.1fs.", name,
                        "chat" if self.is_chat else "encoder-decoder", time.perf_counter() - start)
        except StreamHubModelError:
            raise
        except Exception as exc:
            raise StreamHubModelLoadError(f"Could not initialize model '{name}'") from exc

    def _first_token_ids(self, words):
        ids = {self.tokenizer.encode(word, add_special_tokens=False)[0] for word in words}
        return sorted(ids)

    def _build_prefix_cache(self, system_prompt):
        prefix = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            add_generation_prompt=False, return_tensors="pt", return_dict=True)
        with self.torch.inference_mode():
            output = self.model(**prefix, use_cache=True)
        self.prefix_caches[system_prompt] = (prefix["input_ids"], output.past_key_values)

    def _chat_inputs(self, messages):
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        cached = self.prefix_caches.get(messages[0]["content"])
        if cached is not None:
            prefix_ids, prefix_cache = cached
            n = prefix_ids.shape[-1]
            full = inputs["input_ids"]
            if full.shape[-1] > n and self.torch.equal(full[:, :n], prefix_ids):
                return inputs, copy.deepcopy(prefix_cache), n
        return inputs, None, 0

    def off_topic_probability(self, message, history):
        inputs, past, n_cached = self._chat_inputs(build_scope_messages(message, history))
        with self.torch.inference_mode():
            if past is not None:
                output = self.model(input_ids=inputs["input_ids"][:, n_cached:],
                                    attention_mask=inputs["attention_mask"],
                                    past_key_values=past, use_cache=True)
            else:
                output = self.model(**inputs)
        logits = output.logits[0, -1]
        yes = logits[self.yes_ids].max()
        no = logits[self.no_ids].max()
        return self.torch.softmax(self.torch.stack([yes, no]), dim=0)[1].item()

    def stream(self, message, history, system_prompt=SYSTEM_PROMPT):
        streamer = self.streamer_cls(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        if self.is_chat:
            inputs, past, _ = self._chat_inputs(build_messages(message, history, system_prompt))
            extra = {"past_key_values": past} if past is not None else {}
            yield from _run_streaming(self.model, streamer, **inputs, **extra,
                                      max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      repetition_penalty=1.1, pad_token_id=self.pad_id)
        else:
            inputs = self.tokenizer(build_prompt(message, history), return_tensors="pt",
                                    truncation=True, max_length=512)
            yield from _run_streaming(self.model, streamer, **inputs,
                                      max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      no_repeat_ngram_size=3)


@lru_cache(maxsize=1)
def _get_generator():
    return LocalModel(MODEL_NAME)


def warm_up():
    try:
        _get_generator()
        return True
    except StreamHubModelError:
        logger.exception("Could not load %s", MODEL_NAME)
        return False


def _plain(text):
    return _normalize(text).replace("\u2019", "'")


SUPPORT_TERM_RE = re.compile(
    r"\b(?:streamhub|video|watch\w*|play(?:back|ing)?|buffer\w*|freez\w*|froze|apps?|devices?|"
    r"subscriptions?|subscrib(?:e|es|ed|ing)|plans?|trials?|billing|bill(?:ing|ed|s)?|"
    r"payments?|refunds?|cancel(?:ation|ed|ing|s)?|accounts?|streams?|restart(?:ed|ing|s)?|"
    r"reinstall(?:ed|ing|s)?|internet|wi-?fi|connections?|charg(?:e|es|ed|ing)|"
    r"upgrad(?:e|es|ed|ing)|downgrad(?:e|es|ed|ing)|errors?)\b"
)
_CHAT_PHRASES = (
    r"(?:hi|hello|hey|howdy)(?: there| again)?",
    r"good (?:morning|afternoon|evening)",
    r"(?:thanks|thank you|thx)(?: so much| a lot| again)?"
    r"(?: that'?s all| that helps| that helped| that fixed it)?",
    r"that'?s all(?: thanks| thank you)?",
    r"(?:yes |great |ok |okay )?that (?:fixed it|worked|helped|helps)(?: thanks| thank you)?",
    r"(?:ok|okay|great|perfect|got it|sure|yes|no)(?: thanks| thank you)?",
    r"(?:bye|goodbye)",
    r"(?:it(?:'?s| is)? )?still (?:not working|broken|doesn'?t work|does not work|the same|happening)",
    r"(?:that |it )?(?:didn'?t|did not) (?:help|work)",
    r"(?:it )?(?:doesn'?t|does not) work",
    r"not working",
    r"what else (?:can|should) i (?:try|do)",
    r"what (?:should i|do i) (?:try|do) next",
    r"anything else(?: i can try)?",
)
CHAT_RE = re.compile(r"(?:" + "|".join(_CHAT_PHRASES) + r")")


def is_conversational(message):
    text = re.sub(r"[^\w' ]+", " ", _plain(message))
    return bool(CHAT_RE.fullmatch(" ".join(text.split())))


def is_off_topic(model, message, history):
    if not SCOPE_CHECK or not getattr(model, "is_chat", False):
        return False
    if SUPPORT_TERM_RE.search(_plain(message)) or is_conversational(message):
        return False
    return model.off_topic_probability(message, history) >= OFF_TOPIC_THRESHOLD


_PLAN_WORD_RE = re.compile(r"\b(?:plan|subscription|tier)")
_CHANGE_WORD_RE = re.compile(r"\b(?:chang|upgrad|downgrad|switch)")
_PLAYBACK_TOPIC_RE = re.compile(
    r"\b(?:video|playback|buffer|freez|froze|not playing|won't play|wont play|restart|reinstall|internet|wifi|wi-fi|connection)"
)
_FOLLOW_UP_RE = re.compile(
    r"\b(?:still|already|tried|not working|doesn't work|does not work|didn't help|did not help|didn't work|did not work|no change|same problem|what else|what next|now what|next)\b"
)
_OTHER_TOPIC_RE = re.compile(r"\b(?:refund|billing|bill|trial|cancel|subscription|plan|charge|price|payment)")
_STEP_NEGATION_WINDOW_RE = r"(?:didn'?t|did not|haven'?t|have not|never|not)"


def get_plan_change_policy_response(message):
    text = _plain(message)
    if "prorat" in text or (_PLAN_WORD_RE.search(text) and _CHANGE_WORD_RE.search(text)):
        return PLAN_CHANGE_REPLY
    return None


def _latest_user_step_state(turns, message, stems):
    state = None
    for text in [t["content"] for t in turns if t["role"] == "user"] + [message]:
        plain = _plain(text)
        if not any(re.search(rf"\b{stem}\w*\b", plain) for stem in stems):
            continue
        negated = any(
            re.search(rf"\b{_STEP_NEGATION_WINDOW_RE}\b[^.?!]{{0,30}}\b{stem}\w*\b", plain)
            for stem in stems
        )
        state = not negated
    return state


def get_playback_troubleshooting_response(message, history):
    turns = clean_history(history, limit=None)
    current = _plain(message)
    if _OTHER_TOPIC_RE.search(current):
        return None
    last_reply = next((t["content"] for t in reversed(turns) if t["role"] == "assistant"), "")
    mid_sequence = last_reply in PLAYBACK_REPLIES
    if not (_PLAYBACK_TOPIC_RE.search(current) or (mid_sequence and _FOLLOW_UP_RE.search(current))):
        return None

    given = {t["content"] for t in turns if t["role"] == "assistant"}
    follow_up = bool(_FOLLOW_UP_RE.search(current))
    restart_state = _latest_user_step_state(turns, message, ("restart",))
    internet_state = _latest_user_step_state(turns, message, ("internet", "wifi", "wi-fi", "connection"))
    reinstall_state = _latest_user_step_state(turns, message, ("reinstall",))

    restarted = (restart_state is True) or (restart_state is None and PLAYBACK_RESTART_STEP in given and follow_up)
    checked_internet = (internet_state is True) or (
        internet_state is None and PLAYBACK_INTERNET_STEP in given and follow_up
    )
    reinstalled = (reinstall_state is True) or (
        reinstall_state is None and PLAYBACK_REINSTALL_STEP in given and follow_up
    )
    if not restarted:
        return PLAYBACK_RESTART_STEP
    if not checked_internet:
        return PLAYBACK_INTERNET_STEP
    if not reinstalled:
        return PLAYBACK_REINSTALL_STEP
    return PLAYBACK_DONE


def _rotate_off_topic_reply(message):
    index = int(hashlib.sha256(_normalize(message).encode("utf-8")).hexdigest(), 16) % len(OFF_TOPIC_REPLY_VARIANTS)
    return OFF_TOPIC_REPLY_VARIANTS[index]


def _fixed_conversational_response(message):
    text = " ".join(re.sub(r"[^\w' ]+", " ", _plain(message)).split())
    words = text.split()
    if text in ("thanks", "thank you", "thx", "thanks that's all", "thanks thats all",
                "thank you that's all", "thank you thats all"):
        return "You're welcome. If you need help later, ask about playback, plans, billing, trial, or refunds."
    if len(words) <= 8 and any(
        phrase in text for phrase in ("can you repeat", "repeat that", "say that again", "clarify", "what do you mean")
    ):
        return "Sure. Tell me which part to repeat: playback steps, billing cycle, plan changes, trial, or refunds."
    return None


def _polish_reply(text):
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return cleaned
    cleaned = re.sub(r"\b(\w+(?:\s+\w+){0,6})\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*([?.!])\1+", r"\1", cleaned)
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def factual_override(message):
    text = _normalize(message)
    stream_terms = (
        "simultaneous stream", "simultaneous streams", "stream at the same time", "streams at the same time",
        "devices can stream", "devices at the same time", "how many devices", "can watch at the same time",
        "people can watch", "how many can watch",
    )
    if any(term in text for term in stream_terms):
        return ("StreamHub's exact simultaneous-stream limit is not provided. "
                "I don't have enough information to give you a number.")

    recovery_terms = (
        "when will my video", "when will the video", "when will my playback", "when will playback",
        "when will it start working", "when will it work", "how long until my video", "how long until the video",
        "how long until playback", "how long before my video", "how long before the video",
        "how long before playback",
    )
    if any(term in text for term in recovery_terms):
        return ("StreamHub does not provide a specific recovery time for playback problems. "
                "Try restarting the app/device, then check the internet connection, "
                "and reinstall the app as a last resort.")

    billing_terms = (
        "renewal date", "when is my renewal", "when does my billing renew", "when will i be charged",
        "when will i be billed", "when am i charged", "next billing date", "my next charge",
    )
    if any(term in text for term in billing_terms):
        return ("StreamHub billing is monthly. "
                "The exact renewal date for an individual account is not provided, "
                "and I don't have access to account-specific billing details.")

    trial_terms = ("cancel trial", "cancel my trial", "trial cancellation", "refund trial", "trial refund")
    if any(term in text for term in trial_terms):
        return ("Trial cancellation and trial refund details are not specified in the StreamHub policy. "
                "I can confirm new subscribers get a 7-day free trial.")

    action_terms = (
        "cancel it for me", "change my plan for me", "upgrade my plan for me", "downgrade my plan for me",
        "charge my card", "take payment", "access my account", "transfer me to an agent", "human agent",
    )
    if any(term in text for term in action_terms):
        return ("I can’t access accounts, change plans, take payments, or transfer to an agent. "
                "I can explain StreamHub's playback, plan, trial, billing, and refund policies.")
    return None


def stream_response(message, history):
    if not message.strip():
        yield HELP_TEXT
        return

    checked = guard.check_input(message)
    if checked.blocked:
        logger.info("event=%s kind=%s", EVENT_GUARD_BLOCK, checked.kind)
        yield checked.reply
        return

    conversational = _fixed_conversational_response(message)
    if conversational is not None:
        yield conversational
        return

    override = factual_override(message)
    if override is not None:
        logger.info("event=%s", EVENT_FACTUAL_OVERRIDE)
        yield override
        return

    plan_response = get_plan_change_policy_response(message)
    if plan_response is not None:
        logger.info("event=%s", EVENT_PLAN_OVERRIDE)
        yield plan_response
        return

    playback_response = get_playback_troubleshooting_response(message, history)
    if playback_response is not None:
        logger.info("event=%s", EVENT_PLAYBACK_OVERRIDE)
        yield playback_response
        return

    cached = CACHE.get(message, history)
    if cached is not None:
        logger.info("event=%s", EVENT_CACHE_HIT)
        yield cached
        return

    start = time.perf_counter()
    first_token_at = None
    text = ""
    try:
        model = _get_generator()
        if is_off_topic(model, message, history):
            text = _rotate_off_topic_reply(message)
            logger.info("event=%s", EVENT_OFF_TOPIC)
            CACHE.put(message, history, text)
            yield text
            return
        for chunk in model.stream(message, history):
            if first_token_at is None:
                first_token_at = time.perf_counter() - start
            text += chunk
            yield text
    except StreamHubModelError:
        logger.exception("event=%s", EVENT_MODEL_FAILURE)
        yield FALLBACK_TEXT
        return
    except Exception:
        logger.exception("event=%s", EVENT_MODEL_FAILURE)
        yield FALLBACK_TEXT
        return

    text = text.strip()
    if not text:
        yield HELP_TEXT
        return
    if UNSAFE_OUTPUT_RE.search(text):
        text = SAFE_OUTPUT_REPLY
    else:
        text = _polish_reply(text)

    logger.info("Generated reply: first words after %.1fs, finished in %.1fs.",
                first_token_at or 0.0, time.perf_counter() - start)
    CACHE.put(message, history, text)
    yield text


def response_for(message, history):
    reply = HELP_TEXT
    for reply in stream_response(message, history):
        pass
    return reply


def add_message(message, history):
    history = list(history or [])
    message = (message or "").strip()
    if not message:
        yield history, ""
        return
    prior = list(history)
    shown = prior + [{"role": "user", "content": guard.redact(message)[0]}]
    yield shown + [{"role": "assistant", "content": ""}], ""
    for partial in stream_response(message, prior):
        yield shown + [{"role": "assistant", "content": partial}], ""
