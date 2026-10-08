"""User-safe error messages and logging event labels."""

USER_SAFE_FALLBACK_REPLY = (
    "I'm having trouble reaching the support model right now. "
    "I can still help with playback, subscription timing, free trials, billing cycles, "
    "refund policy, trial cancellation details, and simultaneous-stream limits."
)

EVENT_CACHE_HIT = "cache_hit"
EVENT_GUARD_BLOCK = "guard_block"
EVENT_FACTUAL_OVERRIDE = "factual_override"
EVENT_PLAN_OVERRIDE = "plan_override"
EVENT_PLAYBACK_OVERRIDE = "playback_override"
EVENT_OFF_TOPIC = "off_topic_redirect"
EVENT_MODEL_FAILURE = "model_failure"

