---
title: StreamHub Support
emoji: 🎬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

# StreamHub Support — MSAI HCI Group Project

This repository is a **group project for MSAI 631 (Artificial Intelligence and Human-Computer Interaction)** at the University of the Cumberlands.

It implements a policy-grounded support chatbot for a fictional streaming platform (**StreamHub**) with an accessible Gradio UI. The assistant focuses on:

- playback troubleshooting;
- subscriptions, plan changes, and trial policy;
- billing and refund policy questions.

## What this project does

The chatbot uses a **prompt-only architecture**:

- no fine-tuning;
- no retrieval/database lookup;
- a fixed support policy prompt in `streamhub/service.py`.

Responses are produced by a local open model (default: `Qwen/Qwen2.5-0.5B-Instruct`) and streamed into the chat UI.

## How requests are handled

Each incoming message goes through this flow:

1. **Rule guardrails** (`streamhub/guardrails.py`)  
   - Block prompt-injection attempts.
   - Detect and mask sensitive data (valid card numbers, SSNs, explicit password disclosures).
   - Return fixed safe replies for blocked cases.

2. **Deterministic policy overrides** (`streamhub/service.py`)  
   Some topics bypass generation and return fixed answers to avoid hallucinations, including:
   - plan-change timing/proration;
   - playback step progression (restart → connection check → reinstall);
   - unknown simultaneous-stream limits;
   - unknown playback recovery time;
   - renewal-date/account-action limitations.

3. **Response cache** (`ResponseCache`)  
   Reuses replies for identical normalized message + history + policy/settings/model context.

4. **Scope routing**  
   - Keyword/conversational fast path for obvious in-scope follow-ups.
   - Optional model-based off-topic scoring (`STREAMHUB_SCOPE_CHECK` and threshold).
   - Off-topic requests receive a short redirect reply.

5. **Model generation + output safety**
   - Streams generated output token-by-token.
   - Replaces unsafe generated output (for example, asking for passwords) with a fixed safe reply.

## Repository structure

| Path | Purpose |
|---|---|
| `app.py` | Gradio app entrypoint and accessible UI layout/theme/JS behavior. |
| `streamhub/service.py` | Core orchestration: prompting, fixed-policy routing, scope checks, model streaming, caching. |
| `streamhub/guardrails.py` | Input security/safety checks and off-topic system prompts. |
| `config/settings.py` | Typed environment configuration parsing and validation. |
| `streamhub/errors.py` | User-safe fallback text and logging event constants. |
| `streamhub/exception.py` | Runtime exception hierarchy. |
| `streamhub/model.py` / `streamhub/ui.py` | API re-exports for model and UI layers. |
| `support.py` / `guard.py` | Backward-compatible import shims (deprecated path). |
| `tests/` | Unit test suite (`unittest`) for prompting, routing, safety, caching, and UI contracts. |
| `scripts/evaluate_guard.py` | Evaluates guard/scope behavior on labeled examples using a real chat model. |
| `scripts/accessibility_check.py` | Automated WCAG 2.2 AA checks with Playwright + axe-core. |
| `scripts/benchmark.py` | Quick route/latency benchmark. |
| `docs/ACCESSIBILITY.md` | Accessibility findings, implemented fixes, and manual checks. |
| `docs/VALIDATION.md` | Validation summary and remaining manual validation tasks. |

## Complete run instructions (local)

### 1) Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install required packages

```powershell
python -m pip install -r requirements.txt
```

### 3) Start the app

```powershell
python app.py
```

Open the local Gradio URL shown in terminal (typically `http://127.0.0.1:7860`).

### Notes

- On first run, the selected model is downloaded from Hugging Face.
- If model loading fails, startup logs show the error and the app uses a safe fallback reply path.

## Configuration (environment variables)

Set variables before launching:

```powershell
$env:STREAMHUB_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
python app.py
```

| Variable | Default | Meaning |
|---|---|---|
| `STREAMHUB_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model name passed to Transformers `from_pretrained`. |
| `STREAMHUB_CACHE_FILE` | `.cache/streamhub_responses.json` | Persistent response cache file; empty value disables file persistence. |
| `STREAMHUB_SCOPE_CHECK` | `on` | Enables/disables model-based off-topic classifier logic. |
| `STREAMHUB_OFFTOPIC_THRESHOLD` | `0.65` | Probability threshold for classifying a message as off-topic. |
| `STREAMHUB_MAX_NEW_TOKENS` | `120` | Maximum tokens generated for model answers. |

## Testing and evaluation

### Unit tests (fast, no model download)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Current repository state: **62 tests pass**.

### Guardrail/scope evaluation (uses real model)

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_guard
```

### Accessibility automation (browser-based)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe -m scripts.accessibility_check
```

### Optional benchmark

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark
```

## Accessibility summary

The UI targets **WCAG 2.2 AA** and includes:

- high-contrast controls and visible focus rings;
- keyboard-first navigation with a skip link;
- screen-reader speaker labels;
- live-region handling that announces typing + complete response (not token fragments);
- reduced-motion support;
- responsive layout checks at small width and zoom.

See `docs/ACCESSIBILITY.md` for full evidence and remaining manual checks.

## Known limitations

- Scope detection still depends partly on a small model and can misclassify edge cases.
- Regex-based injection/sensitive-data detection is strong but not exhaustive.
- The assistant intentionally cannot perform account actions (no account access, payments, plan mutations, or agent transfer).
