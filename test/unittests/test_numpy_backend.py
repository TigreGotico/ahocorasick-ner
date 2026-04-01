"""Tests for the numpy and ONNX Aho-Corasick NER backends.

Mirrors test_ner.py test cases to ensure identical behavior.
"""
import os
import tempfile
import unittest

from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER


class TestNumpyAhocorasickNER(unittest.TestCase):
    """Tests for the pure-numpy backend."""

    def setUp(self) -> None:
        self.ner = NumpyAhocorasickNER()
        self.ner.add_word("artist_name", "Metallica")
        self.ner.add_word("artist_name", "Iron Maiden")
        self.ner.add_word("genre", "Heavy Metal")
        self.ner.fit()

    def test_tag(self) -> None:
        text = "I love Metallica and Iron Maiden, they play Heavy Metal."
        tags = list(self.ner.tag(text))
        words = [t["word"] for t in tags]
        labels = [t["label"] for t in tags]

        self.assertIn("Metallica", words)
        self.assertIn("Iron Maiden", words)
        self.assertIn("Heavy Metal", words)

        self.assertEqual(labels[words.index("Metallica")], "artist_name")
        self.assertEqual(labels[words.index("Iron Maiden")], "artist_name")
        self.assertEqual(labels[words.index("Heavy Metal")], "genre")

    def test_word_boundaries(self) -> None:
        self.ner.add_word("instrument", "bass")
        self.ner.fit()

        tags = list(self.ner.tag("He plays the bass.", min_word_len=3))
        self.assertTrue(any(t["word"] == "bass" for t in tags))

        tags = list(self.ner.tag("The embassy is closed.", min_word_len=3))
        self.assertFalse(any(t["word"] == "bass" for t in tags))

    def test_greedy_match(self) -> None:
        self.ner.add_word("genre", "Metal")
        self.ner.add_word("genre", "Heavy Metal")
        self.ner.fit()

        tags = list(self.ner.tag("I listen to Heavy Metal.", min_word_len=3))
        words = [t["word"] for t in tags]
        self.assertIn("Heavy Metal", words)
        self.assertNotIn("Metal", [t["word"] for t in tags if t["word"] != "Heavy Metal"])

    def test_save_load_roundtrip(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            self.ner.save(path)
            loaded = NumpyAhocorasickNER()
            loaded.load(path)

            text = "I love Metallica and Iron Maiden, they play Heavy Metal."
            orig = list(self.ner.tag(text))
            reloaded = list(loaded.tag(text))
            self.assertEqual(orig, reloaded)
        finally:
            os.unlink(path)

    def test_case_insensitive(self) -> None:
        text = "I LOVE METALLICA"
        tags = list(self.ner.tag(text))
        self.assertTrue(any(t["word"] == "METALLICA" for t in tags))

    def test_empty_text(self) -> None:
        tags = list(self.ner.tag(""))
        self.assertEqual(tags, [])

    def test_no_matches(self) -> None:
        tags = list(self.ner.tag("Nothing to see here."))
        self.assertEqual(tags, [])


class TestOnnxAhocorasickNER(unittest.TestCase):
    """Tests for the ONNX backend."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
            cls.onnx_available = True
        except ImportError:
            cls.onnx_available = False

    def setUp(self) -> None:
        if not self.onnx_available:
            self.skipTest("onnx/onnxruntime not installed")
        from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER
        self.ner = OnnxAhocorasickNER()
        self.ner.add_word("artist_name", "Metallica")
        self.ner.add_word("artist_name", "Iron Maiden")
        self.ner.add_word("genre", "Heavy Metal")
        self.ner.fit()

    def test_tag(self) -> None:
        text = "I love Metallica and Iron Maiden, they play Heavy Metal."
        tags = list(self.ner.tag(text))
        words = [t["word"] for t in tags]
        labels = [t["label"] for t in tags]

        self.assertIn("Metallica", words)
        self.assertIn("Iron Maiden", words)
        self.assertIn("Heavy Metal", words)

        self.assertEqual(labels[words.index("Metallica")], "artist_name")
        self.assertEqual(labels[words.index("Iron Maiden")], "artist_name")
        self.assertEqual(labels[words.index("Heavy Metal")], "genre")

    def test_word_boundaries(self) -> None:
        self.ner.add_word("instrument", "bass")
        self.ner.fit()

        tags = list(self.ner.tag("He plays the bass.", min_word_len=3))
        self.assertTrue(any(t["word"] == "bass" for t in tags))

        tags = list(self.ner.tag("The embassy is closed.", min_word_len=3))
        self.assertFalse(any(t["word"] == "bass" for t in tags))

    def test_greedy_match(self) -> None:
        self.ner.add_word("genre", "Metal")
        self.ner.add_word("genre", "Heavy Metal")
        self.ner.fit()

        tags = list(self.ner.tag("I listen to Heavy Metal.", min_word_len=3))
        words = [t["word"] for t in tags]
        self.assertIn("Heavy Metal", words)
        self.assertNotIn("Metal", [t["word"] for t in tags if t["word"] != "Heavy Metal"])

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "model")
            self.ner.save(base)

            from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER
            loaded = OnnxAhocorasickNER()
            loaded.load(base)

            text = "I love Metallica and Iron Maiden, they play Heavy Metal."
            orig = list(self.ner.tag(text))
            reloaded = list(loaded.tag(text))
            self.assertEqual(orig, reloaded)

    def test_case_insensitive(self) -> None:
        text = "I LOVE METALLICA"
        tags = list(self.ner.tag(text))
        self.assertTrue(any(t["word"] == "METALLICA" for t in tags))


if __name__ == "__main__":
    unittest.main()
