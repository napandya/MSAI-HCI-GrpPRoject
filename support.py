"""Compatibility shim: use streamhub.service."""

import sys
import warnings

from streamhub import service as _service

warnings.warn(
    "Importing from 'support' is deprecated; import from 'streamhub.service' instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _service
