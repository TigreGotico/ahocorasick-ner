"""Tests for the pure-Python Aho-Corasick automaton builder."""
import unittest

import numpy as np

from ahocorasick_ner.automaton import AhocorasickAutomaton


class TestAhocorasickAutomaton(unittest.TestCase):
    """Tests for automaton construction and table export."""

    def test_basic_build(self) -> None:
        auto = AhocorasickAutomaton()
        auto.add_word("greeting", "hello")
        auto.add_word("greeting", "hi")
        auto.build()
        tables = auto.to_tables()

        self.assertIn("goto", tables)
        self.assertIn("failure", tables)
        self.assertIn("output", tables)
        self.assertIn("output_counts", tables)
        self.assertIn("char_to_id", tables)
        self.assertIn("label_strings", tables)

    def test_labels_property(self) -> None:
        auto = AhocorasickAutomaton()
        auto.add_word("city", "london")
        auto.add_word("country", "france")
        auto.build()
        self.assertEqual(auto.labels, ["city", "country"])

    def test_to_tables_auto_builds(self) -> None:
        """to_tables() should auto-call build() if not built."""
        auto = AhocorasickAutomaton()
        auto.add_word("test", "hello")
        # Don't call build() explicitly
        tables = auto.to_tables()
        self.assertTrue(tables["goto"].shape[0] > 0)

    def test_failure_link_self_reference(self) -> None:
        """Test edge case where failure link construction handles overlapping patterns."""
        auto = AhocorasickAutomaton()
        # Patterns that create overlapping prefixes
        auto.add_word("a", "abab")
        auto.add_word("b", "bab")
        auto.add_word("c", "ab")
        auto.build()
        tables = auto.to_tables()
        # Should not crash; failure links resolved correctly
        self.assertTrue(tables["failure"].shape[0] > 0)

    def test_output_table_shape(self) -> None:
        auto = AhocorasickAutomaton()
        auto.add_word("x", "abc")
        auto.add_word("y", "bc")
        auto.build()
        tables = auto.to_tables()
        # State for "bc" should have outputs from both "bc" (direct) and
        # possibly merged from failure links
        self.assertEqual(tables["output"].ndim, 3)
        self.assertEqual(tables["output"].shape[2], 2)  # (label_id, match_length)

    def test_empty_automaton(self) -> None:
        auto = AhocorasickAutomaton()
        auto.build()
        tables = auto.to_tables()
        self.assertEqual(tables["goto"].shape[0], 1)  # Just root
        self.assertEqual(int(tables["output_counts"][0]), 0)

    def test_unknown_char_maps_to_zero(self) -> None:
        auto = AhocorasickAutomaton()
        auto.add_word("test", "abc")
        auto.build()
        tables = auto.to_tables()
        # Character 'z' was never added, so its ordinal maps to 0
        char_to_id = tables["char_to_id"]
        z_ord = ord("z")
        if z_ord < len(char_to_id):
            self.assertEqual(int(char_to_id[z_ord]), 0)


if __name__ == "__main__":
    unittest.main()
