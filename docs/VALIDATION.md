# Validation — September 27, 2026

## Done

- The response layer uses the proposal's prompt-only LLM design. The initial candidate,
  `google/flan-t5-base`, tended to echo conversation text, so the default is now
  `Qwen/Qwen2.5-0.5B-Instruct`, which the team confirmed gives better replies in the
  running app.
- Guardrails added: rule checks for instruction-override attempts and sensitive data
  (masked in the chat, never sent to the model), a support keyword and greeting check plus a model-based
  scope check, and a fixed redirect for off-topic questions. Fixed answers cover simultaneous
  streams, playback recovery time, renewal dates, plan changes, trial-policy gaps, unsupported
  account actions, and the playback step order.
- The 62 automated unit tests pass. They replace the model with a stand-in and check the
  surrounding logic: prompts, streaming, caching, the guard rules (including real support
  messages that must not be blocked), prompt-injection hardening, package-module structure,
  exception handling paths, off-topic routing, and chat formatting.
- `python -m scripts.evaluate_guard` has been run with the real model. Current output: app
  accuracy 27/27 (100%), model-only scope accuracy 13/27 (48%), support questions wrongly
  refused by the app 0, and average scope-check time about 693 ms.
- Accessibility audit against WCAG 2.2 AA: nine issues found and fixed, and
  `python -m scripts.accessibility_check` passes all 20 browser checks. Details in `docs/ACCESSIBILITY.md`.
- The prompt prefix cache was checked against a small local test model: it produces the
  same tokens and scope-check scores as processing the full prompt.

## Still to do before submission

- Re-run `python -m scripts.evaluate_guard` only after any guardrail or prompt changes, then update
  these recorded metrics.
- Manual pass in the running app confirming the replies stay grounded in the policy for all
  six scenarios (playback, trial length, billing cycle, plan-change timing, refunds, stream
  limits), plus a few off-topic questions to check the redirects read naturally and never
  answer the question.
- Note any policy violations (for example the model stating a number of simultaneous
  streams) in the design document.
- Manual accessibility checks listed in `docs/ACCESSIBILITY.md`: a screen reader (NVDA or
  Narrator), keyboard-only use of all three scenarios, 200% and 400% zoom, Windows
  contrast themes, and Voice Access.
