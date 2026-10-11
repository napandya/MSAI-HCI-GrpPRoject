import unittest

import streamhub.service as support


class DeterministicPolicyContractTests(unittest.TestCase):
    def test_factual_override_contracts(self):
        cases = {
            "How many devices can stream at the same time?": "exact simultaneous-stream limit is not provided",
            "When will playback start working?": "does not provide a specific recovery time",
            "When is my renewal date?": "exact renewal date for an individual account is not provided",
            "Can I cancel my trial?": "Trial cancellation and trial refund details are not specified",
            "Can you transfer me to a human agent?": "can’t access accounts, change plans, take payments, or transfer to an agent",
        }
        for message, snippet in cases.items():
            with self.subTest(message=message):
                reply = support.factual_override(message)
                self.assertIsNotNone(reply)
                self.assertIn(snippet, reply)

    def test_plan_change_contract(self):
        self.assertEqual(
            support.get_plan_change_policy_response("Will my plan upgrade be prorated?"),
            support.PLAN_CHANGE_REPLY,
        )


if __name__ == "__main__":
    unittest.main()
