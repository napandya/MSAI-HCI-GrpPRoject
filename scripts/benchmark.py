"""Quick local benchmark for response latency and route usage."""

import time
from collections import Counter

import support


CASES = [
    ("How long is the free trial?", []),
    ("What is my billing cycle?", []),
    ("My video keeps buffering", []),
    ("What's the weather in Chicago?", []),
]


def main():
    routes = Counter()
    durations = []
    for message, history in CASES:
        start = time.perf_counter()
        reply = support.response_for(message, history)
        durations.append(time.perf_counter() - start)
        if reply in support.OFF_TOPIC_REPLY_VARIANTS:
            routes["off_topic"] += 1
        elif reply in support.PLAYBACK_REPLIES:
            routes["playback_override"] += 1
        elif "not specified" in reply.lower() or "don't have enough information" in reply.lower():
            routes["factual_override"] += 1
        else:
            routes["model_or_cached"] += 1
    print(f"Cases: {len(CASES)}")
    print(f"Mean latency (s): {sum(durations) / len(durations):.3f}")
    print(f"Route counts: {dict(routes)}")


if __name__ == "__main__":
    main()

