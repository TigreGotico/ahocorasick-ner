import unittest
from ahocorasick_ner import AhocorasickNER

class TestAhocorasickNER(unittest.TestCase):
    def setUp(self):
        self.ner = AhocorasickNER()
        self.ner.add_word("artist_name", "Metallica")
        self.ner.add_word("artist_name", "Iron Maiden")
        self.ner.add_word("genre", "Heavy Metal")
        self.ner.fit()

    def test_tag(self):
        text = "I love Metallica and Iron Maiden, they play Heavy Metal."
        tags = list(self.ner.tag(text))
        
        # tags are dicts with 'start', 'end', 'word', 'label'
        words = [t['word'] for t in tags]
        labels = [t['label'] for t in tags]
        
        self.assertIn("Metallica", words)
        self.assertIn("Iron Maiden", words)
        self.assertIn("Heavy Metal", words)
        
        self.assertEqual(labels[words.index("Metallica")], "artist_name")
        self.assertEqual(labels[words.index("Iron Maiden")], "artist_name")
        self.assertEqual(labels[words.index("Heavy Metal")], "genre")

    def test_word_boundaries(self):
        self.ner.add_word("instrument", "bass")
        self.ner.fit()
        
        text = "He plays the bass."
        tags = list(self.ner.tag(text, min_word_len=3))
        self.assertTrue(any(t['word'] == "bass" for t in tags))
        
        text = "The embassy is closed."
        tags = list(self.ner.tag(text, min_word_len=3))
        self.assertFalse(any(t['word'] == "bass" for t in tags))

    def test_greedy_match(self):
        self.ner.add_word("genre", "Metal")
        self.ner.add_word("genre", "Heavy Metal")
        self.ner.fit()
        
        text = "I listen to Heavy Metal."
        tags = list(self.ner.tag(text, min_word_len=3))
        
        # 'Heavy Metal' should win over 'Metal'
        words = [t['word'] for t in tags]
        self.assertIn("Heavy Metal", words)
        self.assertNotIn("Metal", [t['word'] for t in tags if t['word'] != "Heavy Metal"])

if __name__ == "__main__":
    unittest.main()
