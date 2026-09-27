import unittest
from unittest.mock import patch

import support
from support import (HELP_TEXT, SUPPORT_FACTS, SYSTEM_PROMPT, add_message, build_messages,
                     build_prompt, clean_history, content_text, response_for)

# Gradio 6 stores message content as a list of typed blocks, not a plain string.
GRADIO6_HISTORY = [
    {"role": "user", "content": [{"type": "text", "text": "My video keeps buffering"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Please restart the app."}]},
]


class FakeGenerator:
    """Stands in for the loaded model so tests run without downloading weights."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, message, history):
        self.calls.append((message, list(history)))
        return self.reply


class ContentTextTests(unittest.TestCase):
    def test_plain_string(self):
        self.assertEqual(content_text("  hello "), "hello")

    def test_gradio6_block_list_returns_text_only(self):
        blocks = [{"type": "text", "text": "I restarted my app"}]
        self.assertEqual(content_text(blocks), "I restarted my app")

    def test_none_and_empty(self):
        self.assertEqual(content_text(None), "")
        self.assertEqual(content_text([]), "")


class PromptTests(unittest.TestCase):
    def test_gradio6_history_is_flattened_to_text(self):
        # Regression: raw block lists used to leak into the prompt and get echoed back.
        prompt = build_prompt("It still buffers", GRADIO6_HISTORY)
        self.assertIn("Customer: My video keeps buffering", prompt)
        self.assertIn("Assistant: Please restart the app.", prompt)
        self.assertNotIn("'type'", prompt)
        self.assertNotIn("[{", prompt)

    def test_prompt_includes_policy_and_ends_with_task(self):
        prompt = build_prompt("What is my billing cycle?", [])
        self.assertIn(SUPPORT_FACTS, prompt)
        self.assertIn("Customer's latest message: What is my billing cycle?", prompt)
        self.assertTrue(prompt.rstrip().endswith("using only the policy:"))

    def test_history_is_capped(self):
        history = [{"role": "user", "content": f"turn {i}"} for i in range(20)]
        turns = clean_history(history)
        self.assertEqual(len(turns), support.MAX_HISTORY_TURNS)
        self.assertEqual(turns[-1]["content"], "turn 19")

    def test_chat_messages_have_system_policy_then_history_then_message(self):
        messages = build_messages("It still buffers", GRADIO6_HISTORY)
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[1], {"role": "user", "content": "My video keeps buffering"})
        self.assertEqual(messages[-1], {"role": "user", "content": "It still buffers"})


class ResponseGenerationTests(unittest.TestCase):
    def test_response_uses_model_output(self):
        fake = FakeGenerator("StreamHub bills monthly.")
        with patch.object(support, "_get_generator", return_value=fake):
            self.assertEqual(response_for("What is my billing cycle?", []), "StreamHub bills monthly.")

    def test_empty_message_does_not_call_model(self):
        with patch.object(support, "_get_generator") as mock_get:
            self.assertEqual(response_for("   ", []), HELP_TEXT)
        mock_get.assert_not_called()

    def test_model_failure_falls_back_gracefully(self):
        def boom(message, history):
            raise RuntimeError("no network")

        with patch.object(support, "_get_generator", return_value=boom), \
                self.assertLogs("streamhub", level="ERROR"):
            self.assertIn(HELP_TEXT, response_for("How long is the free trial?", []))

    def test_blank_model_output_falls_back_to_help_text(self):
        with patch.object(support, "_get_generator", return_value=FakeGenerator("  ")):
            self.assertEqual(response_for("hello", []), HELP_TEXT)


class ChatFormattingTests(unittest.TestCase):
    def test_new_message_is_not_duplicated_in_model_input(self):
        # Regression: the new message used to be appended to history before the model call,
        # so it appeared twice in the prompt.
        fake = FakeGenerator("Next, check your internet connection.")
        with patch.object(support, "_get_generator", return_value=fake):
            add_message("It still buffers", GRADIO6_HISTORY)
        message, history_seen = fake.calls[0]
        self.assertEqual(message, "It still buffers")
        self.assertEqual(len(history_seen), 2)

    def test_messages_append_in_chat_format(self):
        with patch.object(support, "_get_generator", return_value=FakeGenerator("7-day free trial.")):
            history, cleared = add_message("How long is the free trial?", [])
        self.assertEqual(cleared, "")
        self.assertEqual([entry["role"] for entry in history], ["user", "assistant"])
        self.assertEqual(history[-1]["content"], "7-day free trial.")

    def test_blank_message_is_ignored(self):
        history, cleared = add_message("   ", [{"role": "user", "content": "hi"}])
        self.assertEqual(len(history), 1)
        self.assertEqual(cleared, "")


if __name__ == "__main__":
    unittest.main()
