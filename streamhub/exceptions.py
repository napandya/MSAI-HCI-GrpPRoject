"""Compatibility module for older imports.

Prefer importing from `streamhub.exception`.
"""

from streamhub.exception import (StreamHubConfigError, StreamHubError, StreamHubModelError,
                                 StreamHubModelInferenceError, StreamHubModelLoadError)

__all__ = [
    "StreamHubError",
    "StreamHubConfigError",
    "StreamHubModelError",
    "StreamHubModelLoadError",
    "StreamHubModelInferenceError",
]
