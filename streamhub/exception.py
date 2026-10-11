"""Custom exception types for StreamHub runtime errors."""


class StreamHubError(Exception):
    """Base error for StreamHub runtime failures."""


class StreamHubConfigError(StreamHubError):
    """Raised when environment configuration is invalid."""


class StreamHubModelError(StreamHubError):
    """Base error raised for model load/inference failures."""


class StreamHubModelLoadError(StreamHubModelError):
    """Raised when the model or tokenizer cannot be initialized."""


class StreamHubModelInferenceError(StreamHubModelError):
    """Raised when model inference or streaming fails."""

