"""Compatibility shim: use streamhub.guardrails."""

import sys
import warnings

from streamhub import guardrails as _guardrails

warnings.warn(
    "Importing from 'guard' is deprecated; import from 'streamhub.guardrails' instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _guardrails
