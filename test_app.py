"""Failure-path tests with fake generation; no model download required."""
import threading
import time
import unittest
from unittest.mock import patch

import torch
import app


class Tokenizer:
    def encode(self, text):
        return text.split()

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.ones((1, 4), dtype=torch.long)}


class Model:
    def __init__(self, behavior):
        self.behavior = behavior

    def generate(self, **kwargs):
        self.behavior(kwargs)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.lock = threading.Lock()
        self.patches = [patch.object(app, "tokenizer", Tokenizer()),
                        patch.object(app, "generation_lock", self.lock)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def test_success_streams_text_and_releases_lock(self):
        def behavior(kwargs):
            kwargs["streamer"].on_finalized_text("Seven days.", stream_end=True)
        with patch.object(app, "model", Model(behavior)):
            self.assertEqual(list(app.respond("Trial?", []))[-1], "Seven days.")
        self.assertTrue(self.lock.acquire(timeout=1))
        self.lock.release()

    def test_worker_failure_returns_retry_instead_of_hanging(self):
        def behavior(kwargs):
            raise RuntimeError("simulated failure")
        with patch.object(app, "model", Model(behavior)), self.assertLogs(app.log, level="ERROR"):
            self.assertIn("try again", list(app.respond("Trial?", []))[-1])
        self.assertTrue(self.lock.acquire(timeout=1))
        self.lock.release()

    def test_timeout_keeps_lock_until_worker_stops(self):
        started = threading.Event()
        release = threading.Event()
        def behavior(kwargs):
            started.set()
            release.wait(2)
        try:
            with patch.object(app, "model", Model(behavior)), patch.object(app, "REQUEST_TIMEOUT", 0.02):
                result = list(app.respond("Trial?", []))
                self.assertTrue(started.is_set())
                self.assertIn("too long", result[-1])
                self.assertTrue(self.lock.locked())
                self.assertIn("another request", list(app.respond("Trial?", []))[-1])
        finally:
            release.set()
        self.assertTrue(self.lock.acquire(timeout=1))
        self.lock.release()


if __name__ == "__main__":
    unittest.main()
