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

# StreamHub Support

A customer-support demonstration for a fictional streaming service, created for MSAI 631.

## How it works

Every reply comes from `google/flan-t5-base` (via Hugging Face `transformers`), not from
hardcoded strings. The bot's scope is kept narrow and consistent by grounding every call in
a fixed system prompt (`INSTRUCTIONS` + `SUPPORT_FACTS` in `support.py`) rather than by
retrieval or fine-tuning — the same prompt-only design described in the project proposal.
Each turn's prompt includes the fixed policy plus the last few conversation turns, so the
model can track playback troubleshooting progress across turns without inventing new facts.

The interface uses a Teams-inspired support layout: a clear top bar, topic shortcuts, familiar chat bubbles, and a focused message composer. It does not claim to be Microsoft Teams or use Microsoft branding.

## Supported questions

- Playback troubleshooting: restart, check the connection, then reinstall.
- Seven-day free trial.
- Monthly billing cycle.
- Plan changes apply at the next billing cycle, without mid-cycle proration.
- Refund limitations.
- Simultaneous streams: explained as plan-dependent; exact limits are intentionally not included.

The app cannot access accounts, process payments, change plans, or accept sensitive information.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open the URL Gradio prints, normally `http://127.0.0.1:7860`.

The first run downloads `google/flan-t5-base` (~250MB) from Hugging Face, so it needs
internet access once; after that it's cached locally and startup is fast. The model loads
at startup, so if anything goes wrong (no network, missing package) the full error prints
in the terminal before the app opens. To try a different seq2seq model, set the
`STREAMHUB_MODEL` environment variable (for example `google/flan-t5-large`).

## Validate

```powershell
python -m unittest discover -v
```

The unit tests mock the model itself (no download or GPU/CPU inference needed to run them),
and check: the prompt correctly includes the fixed policy and recent history, the response
falls back to help text on an empty message or a model failure, and Gradio chat message
formatting. They do not check the *content* of real model output — do a manual pass in the
running app (see `VALIDATION.md`) to confirm flan-t5-base's actual answers stay grounded in
the policy for each of the six supported scenarios.
