# Validation — September 27, 2026

Reverted the response layer back to the proposal's prompt-only LLM design
(`google/flan-t5-base`), after a brief detour to hardcoded keyword matching that was not
what the proposal committed to. The nine automated unit tests pass; they mock the model
itself so they run offline and check the surrounding logic (prompt construction, history
truncation, fallback behavior on an empty message or a model error, chat message
formatting) rather than the content of real model output.

**Not yet done, and needed before submission:**
- Manual pass in the running app confirming flan-t5-base's actual generated answers stay
  grounded in the fixed policy for all six scenarios (playback, trial length, billing cycle,
  plan-change timing, refunds, stream limits) and for at least one out-of-scope / off-topic
  message.
- If flan-t5-base's answers drift, invent facts, or ignore the instructions (a known risk for
  a small instruction-tuned model asked to follow several constraints from one prompt), note
  that in the design document and consider the fallback model mentioned in the proposal.
- Keyboard navigation, screen-reader behavior, and small-screen layout review (carried over
  from the prior validation pass, still outstanding).
