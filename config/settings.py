"""Typed runtime settings loaded from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path

from streamhub.exception import StreamHubConfigError


@dataclass(frozen=True)
class Settings:
    model_name: str
    cache_file: str
    scope_check: bool
    off_topic_threshold: float
    max_new_tokens: int


def _parse_bool(value: str) -> bool:
    return value.strip().lower() not in ("0", "off", "false", "no")


def load_settings() -> Settings:
    model_name = os.environ.get("STREAMHUB_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    cache_file = os.environ.get("STREAMHUB_CACHE_FILE", str(Path(".cache") / "streamhub_responses.json"))
    scope_check = _parse_bool(os.environ.get("STREAMHUB_SCOPE_CHECK", "on"))
    threshold_raw = os.environ.get("STREAMHUB_OFFTOPIC_THRESHOLD", "0.65")
    tokens_raw = os.environ.get("STREAMHUB_MAX_NEW_TOKENS", "120")

    try:
        off_topic_threshold = float(threshold_raw)
    except ValueError as exc:
        raise StreamHubConfigError("STREAMHUB_OFFTOPIC_THRESHOLD must be a float.") from exc
    if not 0.0 <= off_topic_threshold <= 1.0:
        raise StreamHubConfigError("STREAMHUB_OFFTOPIC_THRESHOLD must be between 0 and 1.")

    try:
        max_new_tokens = int(tokens_raw)
    except ValueError as exc:
        raise StreamHubConfigError("STREAMHUB_MAX_NEW_TOKENS must be an integer.") from exc
    if max_new_tokens <= 0:
        raise StreamHubConfigError("STREAMHUB_MAX_NEW_TOKENS must be greater than 0.")

    return Settings(
        model_name=model_name,
        cache_file=cache_file,
        scope_check=scope_check,
        off_topic_threshold=off_topic_threshold,
        max_new_tokens=max_new_tokens,
    )

