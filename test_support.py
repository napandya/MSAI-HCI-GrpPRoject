import unittest
from unittest.mock import patch

import support
from support import add_message, build_prompt, response_for, HELP_TEXT, INSTRUCTIONS, SUPPORT_FACTS


class FakeGenerator:
    """Stands in for the transformers pipeline so tests run without downloading a model."""

    def __init__(self, text):
        self.text = text
        self.last_prompt = None

    def __call__(self, prompt, **kwargs):
        self.last_prompt = prompt
        return [{"generated_text": self.text}]


class PromptTests(unittest.TestCase):
    def test_prompt_includes_policy_instructions_and_message(self):
        prompt = build_prompt("What is my billing cycle?", [])
        self.assertIn(INSTRUCTIONS, prompt)
        self.assertIn(SUPPORT_FACTS, prompt)
        self.assertIn("User: What is my billing cycle?", prompt)
        self.assertTrue(prompt.strip().endswith("Assistant:"))

    def test_prompt_includes_recent_history_in_order(self):
        history = [
            {"role": "user", "content": "My video keeps buffering"},
            {"role": "assistant", "content": "Please restart the app."},
        ]
        prompt = build_prompt("It still buffers", history)
        self.assertIn("User: My video keeps buffering", prompt)
        self.assertIn("Assistant: Please restart the app.", prompt)
        self.assertIn("User: It still buffers", prompt)
        # history entry should come before the new message in the prompt
        self.assertLess(prompt.index("My video keeps buffering"), prompt.index("It still buffers"))

    def test_prompt_caps_history_length(self):
        history = [{"role": "user", "content": f"turn {i}"} for i in range(20)]
        prompt = build_prompt("latest", history)
        self.assertNotIn("turn 0", prompt)
        self.assertIn(f"turn {19}", prompt)


class ResponseGenerationTests(unittest.TestCase):
    def test_response_uses_model_output(self):
        fake = FakeGenerator("StreamHub bills monthly.")
        with patch.object(support, "_get_generator", return_value=fake):
            reply = response_for("What is my billing cycle?", [])
        self.assertEqual(reply, "StreamHub bills monthly.")
        self.assertIn("What is my billing cycle?", fake.last_prompt)

    def test_empty_message_returns_help_text_without_calling_model(self):
        with patch.object(support, "_get_generator") as mock_get:
            reply = response_for("   ", [])
        mock_get.assert_not_called()
        self.assertEqual(reply, HELP_TEXT)

    def test_model_failure_falls_back_gracefully(self):
        def boom(*args, **kwargs):
            raise RuntimeError("no network")

        with patch.object(support, "_get_generator", return_value=boom):
            reply = response_for("How long is the free trial?", [])
        self.assertIn(HELP_TEXT, reply)

    def test_blank_model_output_falls_back_to_help_text(self):
        fake = FakeGenerator("   ")
        with patch.object(support, "_get_generator", return_value=fake):
            reply = response_for("hello", [])
        self.assertEqual(reply, HELP_TEXT)


class ChatFormattingTests(unittest.TestCase):
    def test_messages_append_in_chat_format(self):
        with patch.object(support, "response_for", return_value="7-day free trial."):
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
