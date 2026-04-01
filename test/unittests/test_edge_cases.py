"""Edge case tests to cover remaining uncovered lines across all backends."""
import os
import tempfile
import unittest

from ahocorasick_ner import AhocorasickNER
from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER


class TestNumpyEdgeCases(unittest.TestCase):
    """Cover uncovered lines in numpy_backend.py."""

    def test_auto_fit_on_tag(self) -> None:
        """tag() should auto-fit if not fitted (line 124)."""
        ner = NumpyAhocorasickNER()
        ner.add_word("city", "London")
        # Don't call fit() explicitly
        tags = list(ner.tag("I visited London", min_word_len=3))
        self.assertTrue(any(t["word"] == "London" for t in tags))

    def test_auto_fit_on_save(self) -> None:
        """save() should auto-fit if not fitted (line 75)."""
        ner = NumpyAhocorasickNER()
        ner.add_word("city", "London")
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ner.save(path)  # Should auto-fit
            loaded = NumpyAhocorasickNER()
            loaded.load(path)
            tags = list(loaded.tag("I visited London", min_word_len=3))
            self.assertTrue(any(t["word"] == "London" for t in tags))
        finally:
            os.unlink(path)

    def test_unknown_char_returns_zero(self) -> None:
        """_char_id() should return 0 for characters not in vocabulary (line 109)."""
        ner = NumpyAhocorasickNER()
        ner.add_word("test", "abc")
        ner.fit()
        # Unicode char not in training vocab
        self.assertEqual(ner._char_id("\U0001f600"), 0)

    def test_unicode_text(self) -> None:
        """Tag text containing unicode characters."""
        ner = NumpyAhocorasickNER()
        ner.add_word("city", "tokyo")
        ner.fit()
        tags = list(ner.tag("I love Tokyo! 🎌", min_word_len=3))
        self.assertTrue(any(t["word"] == "Tokyo" for t in tags))

    def test_case_sensitive_mode(self) -> None:
        ner = NumpyAhocorasickNER(case_sensitive=True)
        ner.add_word("city", "London")
        ner.fit()
        # Exact case matches
        self.assertTrue(any(t["word"] == "London" for t in ner.tag("Visit London", min_word_len=3)))
        # Wrong case doesn't match
        self.assertFalse(any(t["word"] == "london" for t in ner.tag("visit london", min_word_len=3)))


class TestOnnxEdgeCases(unittest.TestCase):
    """Cover uncovered lines in onnx_backend.py."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
            cls.available = True
        except ImportError:
            cls.available = False

    def setUp(self) -> None:
        if not self.available:
            self.skipTest("onnx/onnxruntime not installed")

    def test_auto_fit_on_tag(self) -> None:
        from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER
        ner = OnnxAhocorasickNER()
        ner.add_word("city", "London")
        tags = list(ner.tag("I visited London", min_word_len=3))
        self.assertTrue(any(t["word"] == "London" for t in tags))

    def test_auto_fit_on_save(self) -> None:
        from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER
        ner = OnnxAhocorasickNER()
        ner.add_word("city", "London")
        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "model")
            ner.save(base)
            loaded = OnnxAhocorasickNER()
            loaded.load(base)
            tags = list(loaded.tag("I visited London", min_word_len=3))
            self.assertTrue(any(t["word"] == "London" for t in tags))

    def test_unknown_char_returns_zero(self) -> None:
        from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER
        ner = OnnxAhocorasickNER()
        ner.add_word("test", "abc")
        ner.fit()
        self.assertEqual(ner._char_id("\U0001f600"), 0)


class TestCoreEdgeCases(unittest.TestCase):
    """Additional edge cases for the core AhocorasickNER."""

    def test_save_load_roundtrip(self) -> None:
        ner = AhocorasickNER()
        ner.add_word("city", "New York")
        ner.add_word("city", "London")
        ner.fit()

        with tempfile.NamedTemporaryFile(suffix=".ahocorasick", delete=False) as f:
            path = f.name
        try:
            ner.save(path)
            loaded = AhocorasickNER()
            loaded.load(path)
            tags = list(loaded.tag("Visit New York and London", min_word_len=3))
            words = [t["word"] for t in tags]
            self.assertIn("New York", words)
            self.assertIn("London", words)
        finally:
            os.unlink(path)

    def test_auto_fit_on_tag(self) -> None:
        ner = AhocorasickNER()
        ner.add_word("city", "London")
        # Don't call fit()
        tags = list(ner.tag("Visit London", min_word_len=3))
        self.assertTrue(any(t["word"] == "London" for t in tags))


if __name__ == "__main__":
    unittest.main()
