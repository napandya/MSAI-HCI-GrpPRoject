# Local validation — September 27, 2026

Environment: Windows, Python 3.14.4, Gradio 6.28.0, Transformers 5.17.0,
PyTorch 2.14.0+cpu, four PyTorch CPU threads, cached flan-t5-large.

Seven automated tests passed: history preservation, history overflow, oversized
question rejection, non-text filtering, streaming success, worker failure, and
timeout lock ownership. The Gradio interface constructed successfully.

Final warm benchmark (one run; startup excluded):

| Question | First visible text | Completion | Observed answer |
| --- | ---: | ---: | --- |
| How long is the free trial? | 3.59 s | 4.20 s | Seven days |
| Will an upgrade change my bill today? | 3.22 s | 4.90 s | No; next billing cycle, no proration |
| It still freezes (after restarting) | 3.18 s | 3.71 s | Check internet connection |
| How many simultaneous streams can I use? | 3.96 s | 5.50 s | Exact limits unavailable |
| Can you check my latest charge? | 3.20 s | 4.71 s | Cannot access account/payment history |

These five answers matched the core policy expectations. The playback answer
did not ask whether the step helped, despite the prompt requesting that behavior.
Earlier prompt versions produced incorrect billing and stream-limit answers;
worked examples corrected the tested cases. Prompt-only behavior still needs
broader paraphrase, adversarial, and multi-turn evaluation.

No before/after speedup is claimed: the original application was not benchmarked
under matched conditions. Input guidance adds processing work. The output cap,
bounded history, configurable model/thread count, and benchmark make performance
tunable; they do not guarantee a faster answer for every question.

Not verified: a clean dependency installation, browser accessibility, hosting
deployment, simultaneous real users, or performance of flan-t5-base. Automated
failure tests use fake generation; the five benchmark responses use the real model.
