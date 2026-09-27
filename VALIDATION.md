# Validation — September 27, 2026

## Done

- The response layer uses the proposal's prompt-only LLM design. The initial candidate,
  `google/flan-t5-base`, tended to echo conversation text, so the default is now
  `Qwen/Qwen2.5-0.5B-Instruct`, which the team confirmed gives better replies in the
  running app.
- Guardrails added: rule checks for instruction-override attempts and sensitive data
  (masked in the chat, never sent to the model), a model-based scope check, and a
  model-written redirect for off-topic questions.
- The 34 automated unit tests pass. They replace the model with a stand-in and check the
  surrounding logic: prompts, streaming, caching, the guard rules (including real support
  messages that must not be blocked), off-topic routing, and chat formatting.
- Accessibility audit against WCAG 2.2 AA: nine issues found and fixed, and
  `accessibility_check.py` passes all 20 browser checks. Details in `ACCESSIBILITY.md`.
- The prompt prefix cache was checked against a small local test model: it produces the
  same tokens and scope-check scores as processing the full prompt.

## Still to do before submission

- Run `python evaluate_guard.py` with the real model and record its accuracy, the number of
  support questions wrongly refused, and the average scope-check time. If support questions
  are refused, lower `STREAMHUB_OFFTOPIC_THRESHOLD` or refine `guard.SCOPE_SYSTEM_PROMPT`.
- Manual pass in the running app confirming the replies stay grounded in the policy for all
  six scenarios (playback, trial length, billing cycle, plan-change timing, refunds, stream
  limits), plus a few off-topic questions to check the redirects read naturally and never
  answer the question.
- Note any policy violations (for example the model stating a number of simultaneous
  streams) in the design document.
- Manual accessibility checks listed in `ACCESSIBILITY.md`: a screen reader (NVDA or
  Narrator), keyboard-only use of all three scenarios, 200% and 400% zoom, Windows
  contrast themes, and Voice Access.
