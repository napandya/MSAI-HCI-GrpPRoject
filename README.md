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

## What changed

The first version relied entirely on a local language model. It was slow on a CPU and sometimes gave unsupported answers. This version uses guided support flows for the project’s defined scenarios, so each response is immediate, consistent, and traceable to the fictional policy.

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

## Validate

```powershell
python -m unittest discover -v
```

The checks cover direct billing answers, plan timing, stream-limit honesty, playback follow-ups, and Gradio chat message formatting.
