"""Compare warm response latency; inspect answers separately for correctness."""
import json
import time
import app

QUESTIONS = [
    ("How long is the free trial?", []),
    ("Will an upgrade change my bill today?", []),
    ("It still freezes.", [
        {"role": "user", "content": "The app freezes."},
        {"role": "assistant", "content": "Please restart the app. Did that help?"},
    ]),
    ("How many simultaneous streams can I use?", []),
    ("Can you check my latest charge?", []),
]

if __name__ == "__main__":
    app.load_model()
    list(app.respond("How long is the free trial?", []))  # Warm-up, not scored.
    results = []
    for question, history in QUESTIONS:
        started = time.perf_counter()
        first = None
        answer = ""
        for answer in app.respond(question, history):
            if first is None and answer.strip():
                first = time.perf_counter() - started
        results.append({"question": question, "first_visible_seconds": first,
                        "total_seconds": time.perf_counter() - started, "answer": answer})
    print(json.dumps({"model": app.MODEL_NAME, "results": results}, indent=2))
