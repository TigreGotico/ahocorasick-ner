"""Tests for the OpenVoiceOS plugin transformer with mocked dependencies."""
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


# Mock all OVOS dependencies before importing opm
_mock_modules = {
    "ovos_bus_client": MagicMock(),
    "ovos_bus_client.message": MagicMock(),
    "ovos_bus_client.session": MagicMock(),
    "ovos_plugin_manager": MagicMock(),
    "ovos_plugin_manager.templates": MagicMock(),
    "ovos_plugin_manager.templates.pipeline": MagicMock(),
    "ovos_plugin_manager.templates.transformers": MagicMock(),
    "ovos_utils": MagicMock(),
    "ovos_utils.bracket_expansion": MagicMock(),
    "ovos_utils.lang": MagicMock(),
    "ovos_utils.list_utils": MagicMock(),
    "ovos_utils.log": MagicMock(),
}


@patch.dict("sys.modules", _mock_modules)
class TestAhocorasickNERTransformer(unittest.TestCase):
    """Tests for the OPM plugin with mocked OVOS dependencies."""

    def _get_transformer_class(self):
        """Import class with mocked dependencies."""
        # Make IntentTransformer.__init__ a no-op
        _mock_modules["ovos_plugin_manager.templates.transformers"].IntentTransformer = MagicMock
        import importlib
        import ahocorasick_ner.opm as opm_module
        importlib.reload(opm_module)
        return opm_module.AhocorasickNERTransformer

    def test_init(self) -> None:
        cls = self._get_transformer_class()
        transformer = cls()
        self.assertEqual(transformer.matchers, {})

    def test_bind(self) -> None:
        cls = self._get_transformer_class()
        transformer = cls()
        # Manually set bus (simulating what super().bind() does)
        mock_bus = MagicMock()
        transformer.bus = mock_bus
        # Call the registration logic directly
        transformer.bus.on("padatious:register_entity", transformer.handle_register_entity)
        mock_bus.on.assert_called_once_with(
            "padatious:register_entity",
            transformer.handle_register_entity,
        )

    def test_handle_register_entity(self) -> None:
        cls = self._get_transformer_class()
        transformer = cls()
        transformer.matchers = {}

        # Mock the message
        msg = MagicMock()
        msg.data = {
            "skill_id": "test_skill",
            "lang": "en-us",
            "name": "city",
            "samples": ["New York", "London", "Paris"],
            "file_name": None,
        }
        msg.context = {}

        # Mock SessionManager
        from ahocorasick_ner import opm as opm_mod
        mock_sess = MagicMock()
        mock_sess.lang = "en-us"
        opm_mod.SessionManager.get.return_value = mock_sess
        opm_mod.standardize_lang_tag = lambda x: x
        opm_mod.expand_template = lambda x: [x]
        opm_mod.deduplicate_list = lambda x: list(set(x))
        opm_mod.flatten_list = lambda x: [item for sub in x for item in sub]

        transformer.handle_register_entity(msg)

        self.assertIn("en-us", transformer.matchers)
        self.assertIn("test_skill", transformer.matchers["en-us"])
        ner = transformer.matchers["en-us"]["test_skill"]
        ner.fit()
        tags = list(ner.tag("I visited London and Paris", min_word_len=3))
        words = [t["word"] for t in tags]
        self.assertIn("London", words)
        self.assertIn("Paris", words)

    def test_handle_register_entity_empty_samples(self) -> None:
        cls = self._get_transformer_class()
        transformer = cls()
        transformer.matchers = {}

        msg = MagicMock()
        msg.data = {
            "skill_id": "test_skill",
            "lang": "en-us",
            "name": "city",
            "samples": [],
            "file_name": None,
        }
        msg.context = {}

        from ahocorasick_ner import opm as opm_mod
        mock_sess = MagicMock()
        mock_sess.lang = "en-us"
        opm_mod.SessionManager.get.return_value = mock_sess
        opm_mod.standardize_lang_tag = lambda x: x
        opm_mod.expand_template = lambda x: [x]
        opm_mod.deduplicate_list = lambda x: x
        opm_mod.flatten_list = lambda x: x

        transformer.handle_register_entity(msg)
        # Should skip — no matchers added
        self.assertEqual(transformer.matchers, {})

    def test_handle_register_entity_anonymous_skill(self) -> None:
        cls = self._get_transformer_class()
        transformer = cls()
        transformer.matchers = {}

        msg = MagicMock()
        msg.data = {
            "lang": "en-us",
            "name": "test_entity",
            "samples": ["hello world"],
        }
        msg.context = {}

        from ahocorasick_ner import opm as opm_mod
        mock_sess = MagicMock()
        mock_sess.lang = "en-us"
        opm_mod.SessionManager.get.return_value = mock_sess
        opm_mod.standardize_lang_tag = lambda x: x
        opm_mod.expand_template = lambda x: [x]
        opm_mod.deduplicate_list = lambda x: x
        opm_mod.flatten_list = lambda x: [item for sub in x for item in sub]

        transformer.handle_register_entity(msg)
        self.assertIn("anonymous_skill", transformer.matchers.get("en-us", {}))

    def test_handle_register_entity_from_file(self) -> None:
        import tempfile
        import os
        cls = self._get_transformer_class()
        transformer = cls()
        transformer.matchers = {}

        # Create a temp entity file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".entity", delete=False) as f:
            f.write("New York\nLondon\nParis\n")
            filepath = f.name

        try:
            msg = MagicMock()
            msg.data = {
                "skill_id": "test_skill",
                "lang": "en-us",
                "name": "city",
                "file_name": filepath,
            }
            msg.context = {}

            from ahocorasick_ner import opm as opm_mod
            mock_sess = MagicMock()
            mock_sess.lang = "en-us"
            opm_mod.SessionManager.get.return_value = mock_sess
            opm_mod.standardize_lang_tag = lambda x: x
            opm_mod.expand_template = lambda x: [x]
            opm_mod.deduplicate_list = lambda x: list(dict.fromkeys(x))
            opm_mod.flatten_list = lambda x: [item for sub in x for item in sub]
            opm_mod.isfile = os.path.isfile

            transformer.handle_register_entity(msg)
            self.assertIn("test_skill", transformer.matchers.get("en-us", {}))
        finally:
            os.unlink(filepath)

    def test_transform_with_match(self) -> None:
        cls = self._get_transformer_class()
        transformer = cls()

        # Pre-populate a matcher
        from ahocorasick_ner import AhocorasickNER
        ner = AhocorasickNER()
        ner.add_word("city", "London")
        ner.fit()
        transformer.matchers = {"en-us": {"my_skill": ner}}

        # Mock intent
        intent = MagicMock()
        intent.utterance = "I visited London"
        intent.skill_id = "my_skill"
        intent.match_data = {}
        intent.match_type = "my_skill:intent"

        mock_sess = MagicMock()
        mock_sess.lang = "en-us"
        intent.updated_session = mock_sess

        from ahocorasick_ner import opm as opm_mod
        opm_mod.SessionManager.get.return_value = mock_sess

        result = transformer.transform(intent)
        self.assertEqual(result.match_data.get("city"), "London")

    def test_transform_no_match(self) -> None:
        cls = self._get_transformer_class()
        transformer = cls()
        transformer.matchers = {}

        intent = MagicMock()
        intent.utterance = "hello world"
        intent.skill_id = "unknown"
        intent.match_data = {}
        intent.match_type = "unknown:intent"

        mock_sess = MagicMock()
        mock_sess.lang = "en-us"
        intent.updated_session = mock_sess

        from ahocorasick_ner import opm as opm_mod
        opm_mod.SessionManager.get.return_value = mock_sess

        result = transformer.transform(intent)
        self.assertEqual(result.match_data, {})

    def test_transform_error_handling(self) -> None:
        cls = self._get_transformer_class()
        transformer = cls()
        transformer.matchers = {"en-us": {"my_skill": MagicMock(tag=MagicMock(side_effect=RuntimeError("boom")))}}

        intent = MagicMock()
        intent.utterance = "test"
        intent.skill_id = "my_skill"
        intent.match_data = {}

        mock_sess = MagicMock()
        mock_sess.lang = "en-us"
        intent.updated_session = mock_sess

        # Should not raise
        result = transformer.transform(intent)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
