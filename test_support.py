import unittest
from support import add_message, response_for


class SupportTests(unittest.TestCase):
    def test_billing_cycle_has_a_direct_answer(self):
        self.assertIn("monthly", response_for("tell me what is my billing cycle", []).lower())

    def test_followup_advances_playback_step(self):
        history = [{"role": "user", "content": "The video freezes"}, {"role": "assistant", "content": "Restart the app."}]
        self.assertIn("internet", response_for("It still freezes", history).lower())

    def test_plan_change_is_not_immediate(self):
        self.assertIn("next billing cycle", response_for("Can I upgrade today?", []).lower())

    def test_unknown_stream_limit_is_not_invented(self):
        self.assertIn("does not include", response_for("How many devices can stream?", []).lower())

    def test_messages_append_in_chat_format(self):
        history, cleared = add_message("How long is the free trial?", [])
        self.assertEqual(cleared, "")
        self.assertEqual([entry["role"] for entry in history], ["user", "assistant"])


if __name__ == "__main__":
    unittest.main()
