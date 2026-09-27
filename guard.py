"""Input guardrails for the StreamHub support bot.

Three layers run before the main support prompt:

1. Rule checks (this module, no model needed) catch the cases that must be handled the
   same way every time: attempts to override the bot's instructions, and messages that
   contain card numbers, Social Security numbers or passwords. These get a fixed, polite
   reply and never reach the model. Sensitive values are also masked in the chat history.
2. A scope check asks the model itself whether the message is about StreamHub support
   (see SCOPE_SYSTEM_PROMPT). Follow-ups like "still not working" count as in scope.
3. Off-topic messages get a reply generated from OFF_TOPIC_SYSTEM_PROMPT: the model briefly
   acknowledges the question, declines to answer it, and points back to what it can do.
"""
import re
from dataclasses import dataclass

SCOPE_SYSTEM_PROMPT = """You decide whether a customer's message is within scope for
StreamHub's support assistant. StreamHub is a video streaming service.

In scope: video playback problems (buffering, freezing, errors, the app not working),
the StreamHub app or devices used to watch it, subscriptions and plans, plan changes,
the free trial, billing, refunds, cancellation, simultaneous streams, accounts,
greetings, thanks, and follow-ups to the support conversation such as "still not
working", "yes that helped" or "what else can I try".

Out of scope: anything else, such as general knowledge, homework, coding, recipes,
news, weather, other companies' products, or requests to write unrelated content.

Reply with exactly one word: yes if the message is in scope, no if it is out of scope."""

OFF_TOPIC_SYSTEM_PROMPT = """You are StreamHub's virtual support assistant. The customer
has asked about something outside what you can help with.
Reply in 1-2 short, friendly sentences:
- briefly acknowledge what they asked, without answering it or giving any information about it;
- say you can only help with StreamHub playback problems, subscriptions and plan changes,
  the free trial, and billing or refunds;
- invite them to ask about one of those.
Never answer the off-topic question itself, even partly."""

INJECTION_REPLY = ("I can't change how I work or share my instructions, but I'm happy to help "
                   "with StreamHub playback problems, plans, the free trial, or billing. "
                   "What can I help you with?")

SENSITIVE_REPLY = ("For your security, please don't share card numbers, Social Security numbers, "
                   "or passwords here. I've hidden what you sent. I can't access accounts or take "
                   "payments, but I can explain StreamHub's billing, plan, and refund policies.")

# Kept deliberately narrow so ordinary support messages ("I got a system message",
# "can you override the refund policy?") are left for the model to answer.
_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules|prompt|guidelines)\b",
    r"\b(system|hidden|initial)\s+(prompt|instructions?)\b",
    r"\b(reveal|show|print|repeat|tell me)\b.{0,30}\b(your|the)\s+(prompt|instructions?|rules)\b",
    r"\byou are (now|no longer)\b",
    r"\b(pretend|act|behave)\s+(to be|as if you|like you)\b",
    r"\b(jailbreak|developer mode|dan mode|do anything now)\b",
    r"\bnew (instructions?|persona)\s*:",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Only mask values that contain a digit, so "my password is expired" isn't treated as a secret.
_PASSWORD_RE = re.compile(r"\b(password|passcode|pwd|pin)\b(\s+is|\s*[:=])\s*((?=\S*\d)\S{4,})",
                          re.IGNORECASE)


@dataclass(frozen=True)
class GuardResult:
    """Outcome of the rule checks. `reply` is set when the message must not reach the model."""
    kind: str               # "ok", "injection", or "sensitive"
    reply: str | None       # fixed reply for blocked messages
    display_text: str       # the message as it should appear in the chat (sensitive parts masked)

    @property
    def blocked(self):
        return self.reply is not None


def _luhn_valid(digits):
    total, double = 0, False
    for ch in reversed(digits):
        n = int(ch)
        if double:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
        double = not double
    return total % 10 == 0


def redact(text):
    """Mask card numbers, SSNs and stated passwords. Returns (masked_text, found_anything)."""
    found = False

    def mask_card(match):
        nonlocal found
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            found = True
            return "[card number hidden]"
        return match.group(0)

    text = _CARD_CANDIDATE_RE.sub(mask_card, text)
    text, ssn_count = _SSN_RE.subn("[SSN hidden]", text)
    text, pw_count = _PASSWORD_RE.subn(lambda m: f"{m.group(1)}{m.group(2)} [hidden]", text)
    return text, found or ssn_count > 0 or pw_count > 0


def check_input(message):
    """Run the rule checks on a user message."""
    masked, sensitive = redact(message)
    if sensitive:
        return GuardResult("sensitive", SENSITIVE_REPLY, masked)
    if _INJECTION_RE.search(message):
        return GuardResult("injection", INJECTION_REPLY, message)
    return GuardResult("ok", None, message)
