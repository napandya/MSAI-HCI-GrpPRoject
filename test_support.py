import inspect
import os
import tempfile
import unittest
from unittest.mock import patch

import support
from support import (FALLBACK_TEXT, HELP_TEXT, SUPPORT_FACTS, SYSTEM_PROMPT, ResponseCache,
                     add_message, build_messages, build_prompt, clean_history, content_text,
                     response_for, stream_response)

# Gradio 6 stores message content as a list of typed blocks, not a plain string.
GRADIO6_HISTORY = [
    {"role": "user", "content": [{"type": "text", "text": "My video keeps buffering"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Please restart the app."}]},
]


class FakeModel:
    """Stands in for the loaded model: streams a fixed reply word by word."""

    def __init__(self, reply, fail=False):
        self.reply = reply
        self.fail = fail
        self.calls = []

    def stream(self, message, history):
        self.calls.append((message, list(history)))
        if self.fail:
            raise RuntimeError("no network")
        for word in self.reply.split(" "):
            yield word + " "


class IsolatedCacheTest(unittest.TestCase):
    """Gives each test a fresh in-memory response cache."""

    def setUp(self):
        patcher = patch.object(support, "CACHE", ResponseCache(None))
        patcher.start()
        self.addCleanup(patcher.stop)

    def use_model(self, model):
        patcher = patch.object(support, "_get_generator", return_value=model)
        patcher.start()
        self.addCleanup(patcher.stop)
        return model


class ContentTextTests(unittest.TestCase):
    def test_plain_string(self):
        self.assertEqual(content_text("  hello "), "hello")

    def test_gradio6_block_list_returns_text_only(self):
        self.assertEqual(content_text([{"type": "text", "text": "I restarted my app"}]),
                         "I restarted my app")

    def test_none_and_empty(self):
        self.assertEqual(content_text(None), "")
        self.assertEqual(content_text([]), "")


class PromptTests(unittest.TestCase):
    def test_gradio6_history_is_flattened_to_text(self):
        # Regression: raw block lists used to leak into the prompt and get echoed back.
        prompt = build_prompt("It still buffers", GRADIO6_HISTORY)
        self.assertIn("Customer: My video keeps buffering", prompt)
        self.assertNotIn("'type'", prompt)

    def test_prompt_includes_policy_and_ends_with_task(self):
        prompt = build_prompt("What is my billing cycle?", [])
        self.assertIn(SUPPORT_FACTS, prompt)
        self.assertTrue(prompt.rstrip().endswith("using only the policy:"))

    def test_history_is_capped(self):
        turns = clean_history([{"role": "user", "content": f"turn {i}"} for i in range(20)])
        self.assertEqual(len(turns), support.MAX_HISTORY_TURNS)
        self.assertEqual(turns[-1]["content"], "turn 19")

    def test_chat_messages_have_system_policy_then_history_then_message(self):
        messages = build_messages("It still buffers", GRADIO6_HISTORY)
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[1], {"role": "user", "content": "My video keeps buffering"})
        self.assertEqual(messages[-1], {"role": "user", "content": "It still buffers"})


class StreamingTests(IsolatedCacheTest):
    def test_reply_grows_word_by_word(self):
        self.use_model(FakeModel("StreamHub bills monthly."))
        parts = list(stream_response("What is my billing cycle?", []))
        self.assertGreater(len(parts), 2)
        self.assertTrue(parts[1].startswith(parts[0].strip()))
        self.assertEqual(parts[-1], "StreamHub bills monthly.")

    def test_empty_message_does_not_call_model(self):
        model = self.use_model(FakeModel("unused"))
        self.assertEqual(response_for("   ", []), HELP_TEXT)
        self.assertEqual(model.calls, [])

    def test_model_failure_falls_back_gracefully(self):
        self.use_model(FakeModel("", fail=True))
        with self.assertLogs("streamhub", level="ERROR"):
            self.assertEqual(response_for("How long is the free trial?", []), FALLBACK_TEXT)

    def test_blank_model_output_falls_back_to_help_text(self):
        self.use_model(FakeModel(" "))
        self.assertEqual(response_for("hello", []), HELP_TEXT)


class ResponseCacheTests(IsolatedCacheTest):
    def test_repeat_question_is_answered_from_cache(self):
        model = self.use_model(FakeModel("New subscribers get a 7-day free trial."))
        first = response_for("How long is the free trial?", [])
        second = response_for("how long is the free trial", [])  # case/punctuation differ
        self.assertEqual(first, second)
        self.assertEqual(len(model.calls), 1)

    def test_different_conversation_is_not_a_cache_hit(self):
        model = self.use_model(FakeModel("Next, check your internet connection."))
        response_for("It still does not work", [])
        response_for("It still does not work", GRADIO6_HISTORY)
        self.assertEqual(len(model.calls), 2)

    def test_failures_are_not_cached(self):
        self.use_model(FakeModel("", fail=True))
        with self.assertLogs("streamhub", level="ERROR"):
            response_for("What is my billing cycle?", [])
        self.assertEqual(len(support.CACHE), 0)

    def test_cache_survives_restart_via_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.json")
            ResponseCache(path).put("What is my billing cycle?", [], "Billing is monthly.")
            reloaded = ResponseCache(path)
            self.assertEqual(reloaded.get("what is my billing cycle", []), "Billing is monthly.")

    def test_changing_the_model_invalidates_entries(self):
        cache = ResponseCache(None)
        cache.put("What is my billing cycle?", [], "Billing is monthly.")
        with patch.object(support, "MODEL_NAME", "some/other-model"):
            self.assertIsNone(cache.get("What is my billing cycle?", []))


class ChatHandlerTests(IsolatedCacheTest):
    def test_user_message_shows_immediately_then_reply_streams(self):
        self.use_model(FakeModel("Plan changes apply next billing cycle."))
        updates = list(add_message("When does a plan change apply?", []))
        first_history, cleared = updates[0]
        self.assertEqual(cleared, "")
        self.assertEqual(first_history[-2], {"role": "user", "content": "When does a plan change apply?"})
        self.assertEqual(first_history[-1]["content"], "")
        self.assertEqual(updates[-1][0][-1]["content"], "Plan changes apply next billing cycle.")

    def test_new_message_is_not_duplicated_in_model_input(self):
        # Regression: the new message used to be in both the history and the message.
        model = self.use_model(FakeModel("Next, check your internet connection."))
        list(add_message("It still buffers", GRADIO6_HISTORY))
        message, history_seen = model.calls[0]
        self.assertEqual(message, "It still buffers")
        self.assertEqual(len(history_seen), 2)

    def test_blank_message_is_ignored(self):
        updates = list(add_message("   ", [{"role": "user", "content": "hi"}]))
        self.assertEqual(len(updates[-1][0]), 1)

    def test_quick_topic_handler_streams(self):
        import app
        self.assertTrue(inspect.isgeneratorfunction(app.choose))


if __name__ == "__main__":
    unittest.main()
