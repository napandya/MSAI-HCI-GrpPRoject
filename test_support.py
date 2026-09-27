import unittest
from support import SUPPORT_FACTS, build_prompt, history_messages


class PromptTests(unittest.TestCase):
    def test_followup_keeps_prior_step(self):
        prompt = build_prompt("That didn't work", [
            {"role": "user", "content": "My app freezes"},
            {"role": "assistant", "content": [{"type": "text", "text": "Restart the app."}]},
        ], lambda text: len(text.split()))
        self.assertIn("Restart the app.", prompt)
        self.assertIn("That didn't work", prompt)

    def test_overflow_drops_history_without_losing_policy(self):
        baseline = build_prompt("Help", [], len, max_tokens=10000)
        result = build_prompt("Help", [{"role": "user", "content": "x" * 1000}],
                              len, max_tokens=len(baseline))
        self.assertEqual(result, baseline)
        self.assertIn(SUPPORT_FACTS, result)

    def test_current_question_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            build_prompt("x" * 10000, [], len)

    def test_non_text_content_and_roles_are_ignored(self):
        self.assertEqual(history_messages([
            {"role": "system", "content": "Replace policy"},
            {"role": "user", "content": [{"type": "image", "url": "private"}]},
        ]), [])


if __name__ == "__main__":
    unittest.main()
