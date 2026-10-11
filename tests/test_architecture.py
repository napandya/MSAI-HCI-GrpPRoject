import unittest
from unittest.mock import patch

import guard
import support
import streamhub.guardrails as guardrails
import streamhub.service as service
from config.settings import Settings, load_settings
from streamhub.exception import StreamHubConfigError, StreamHubModelLoadError


class PackageStructureTests(unittest.TestCase):
    def test_support_module_is_service_shim(self):
        self.assertIs(support, service)

    def test_guard_module_is_guardrails_shim(self):
        self.assertIs(guard, guardrails)


class ExceptionHandlingTests(unittest.TestCase):
    def test_warm_up_returns_false_on_model_load_error(self):
        with patch.object(service, "_get_generator", side_effect=StreamHubModelLoadError("load failed")):
            self.assertFalse(service.warm_up())

    def test_stream_response_returns_fallback_on_model_error(self):
        with patch.object(service, "_get_generator", side_effect=StreamHubModelLoadError("load failed")):
            self.assertEqual(service.response_for("How long is the free trial?", []), service.FALLBACK_TEXT)

    def test_fallback_does_not_leak_internal_exception_text(self):
        with patch.object(service, "_get_generator", side_effect=StreamHubModelLoadError("traceback details")):
            reply = service.response_for("How long is the free trial?", [])
            self.assertNotIn("traceback", reply.lower())


class SettingsTests(unittest.TestCase):
    def test_settings_load_default_shape(self):
        settings = load_settings()
        self.assertIsInstance(settings, Settings)
        self.assertGreater(settings.max_new_tokens, 0)
        self.assertTrue(0 <= settings.off_topic_threshold <= 1)

    def test_invalid_threshold_raises(self):
        with patch.dict("os.environ", {"STREAMHUB_OFFTOPIC_THRESHOLD": "2.5"}, clear=False):
            with self.assertRaises(StreamHubConfigError):
                load_settings()


if __name__ == "__main__":
    unittest.main()
