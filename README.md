---
title: StreamHub Support Bot
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

# StreamHub Support Bot

MSAI 631: Artificial Intelligence and Human-Computer Interaction.
A customer-support demonstration for a fictional streaming service.

## Scope and limitations

The bot explains subscription and billing policies and guides playback
troubleshooting. It cannot access accounts, change plans, process payments,
or transfer users to a support agent. Do not enter private account information.

Answers are generated using fixed policy facts and a bounded recent conversation
in the prompt. There is no retrieval or fine-tuning. Prompt instructions do not
guarantee factual accuracy; evaluate generated responses before submission.
Conversation history provides context, not an authoritative source of policies.

All plan changes take effect next billing cycle, with no mid-cycle proration.
Exact plan prices and stream limits are deliberately unspecified; the bot should
say it does not have those details. Trial refund/cancellation terms are also
unspecified. This corrects the original contradictory proration rule without
inventing additional policies.

## Run locally

Use Python 3.11 or 3.12 in a separate environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio. The first launch downloads the model and
requires internet access. Loading happens once at startup, not once per message.
Gradio and Transformers are pinned to the versions used for local validation.
PyTorch and SentencePiece have bounded ranges; this is not a complete lockfile.
Validation used Python 3.14.4, Gradio 6.28.0, Transformers 5.17.0, and CPU
PyTorch 2.14.0 on Windows. A clean Python 3.11/3.12 install was not tested.

## Performance settings

The default remains `google/flan-t5-large`, preserving the original project's
quality choice. CPU latency depends on hardware and output length. Streaming
shows progress; it does not reduce the computation needed for an answer.

- `STREAMHUB_MODEL`: checkpoint; use `google/flan-t5-base` to compare speed/quality.
- `STREAMHUB_MAX_NEW_TOKENS`: output cap, default 60; long answers may be cut short.
- `STREAMHUB_CPU_THREADS`: optional positive PyTorch thread count. Benchmark it;
  more threads are not necessarily faster. Unset uses PyTorch's default.
- `STREAMHUB_TIMEOUT`: generation deadline in seconds, default 90. Cancellation
  takes effect between generation steps, not in the middle of a model operation.

```powershell
$env:STREAMHUB_MODEL = "google/flan-t5-base"
python benchmark.py
```

Repeat with `google/flan-t5-large`. The benchmark warms up the model, then reports
time to first visible output, total latency, and answers for five scenarios.
Review answer correctness alongside timings. No speedup is claimed without a
measurement on the target laptop or hosting environment.

The app uses greedy decoding, inference mode, caching, a shorter output cap,
and at most four recent messages within a 512-token input budget. Oversized
questions are rejected rather than silently removing policies. One request
generates at a time to avoid overlapping CPU work. A timed-out or cancelled
worker retains its lock until it stops. Logs contain timings, not question text.

## CUX evaluation

Run automated checks with `python -m unittest discover -v`.
Manually test these conversations with the actual model:

| Scenario | Expected behavior |
| --- | --- |
| Restart suggested, then “That did not work” | Advance to checking the connection |
| User already tried every available step | Explain limit and summarize; no loop |
| “Upgrade my plan now” | Explain next-cycle policy and inability to act |
| Ask an exact stream limit | Acknowledge that the policy does not specify it |
| Ask an unrelated question | Explain supported topics |
| Frustrated user | Brief acknowledgment and relevant next step |
| Long question | Request a shorter question without losing policy facts |

Also check keyboard-only use, screen-reader labels, contrast, and the Stop/clear
controls. Bounded history is not a guarantee of perfect recall or instruction
following. Record response accuracy, repetitions, latency, and user feedback.

Microsoft references: [CUX principles](https://learn.microsoft.com/en-us/microsoft-copilot-studio/guidance/cux-principles)
and [CUX guide](https://learn.microsoft.com/en-us/azure/bot-service/bot-service-design-principles?view=azure-bot-service-4.0).

## Hosting

This repository contains the application source. To host it, create a Gradio
Hugging Face Space and copy/sync these files. The metadata above declares the
Space entry point; it does not deploy the app automatically. Check memory and
latency on the selected hosting hardware before claiming it meets requirements.
