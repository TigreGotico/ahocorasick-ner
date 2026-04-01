"""Tests for version module."""
import unittest


class TestVersion(unittest.TestCase):
    """Tests for version string construction."""

    def test_version_format(self) -> None:
        from ahocorasick_ner.version import __version__
        self.assertIsInstance(__version__, str)
        # Should be semver with optional alpha
        parts = __version__.split("a")
        self.assertTrue(len(parts) in (1, 2))
        major_minor_patch = parts[0].split(".")
        self.assertEqual(len(major_minor_patch), 3)

    def test_version_components(self) -> None:
        from ahocorasick_ner.version import (
            VERSION_MAJOR, VERSION_MINOR, VERSION_BUILD, VERSION_ALPHA,
        )
        self.assertIsInstance(VERSION_MAJOR, int)
        self.assertIsInstance(VERSION_MINOR, int)
        self.assertIsInstance(VERSION_BUILD, int)


if __name__ == "__main__":
    unittest.main()
