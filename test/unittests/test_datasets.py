"""Tests for dataset loader classes with mocked HuggingFace datasets."""
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from ahocorasick_ner.datasets import EncyclopediaMetallvmNER, MusicNER, ImdbNER


def _make_hf_dataset(entries):
    """Creates a mock HF dataset that behaves like dataset['train']."""
    mock_ds = MagicMock()
    mock_ds.__getitem__ = lambda self, key: entries if key == "train" else []
    return mock_ds


class TestEncyclopediaMetallvmNER(unittest.TestCase):
    """Tests for the Encyclopedia Metallum NER loader."""

    @patch("ahocorasick_ner.datasets.load_dataset")
    def test_load_huggingface(self, mock_load: MagicMock) -> None:
        tracks_data = [
            {"band_name": "Metallica", "track_name": "Fade to Black",
             "album_name": "Ride the Lightning", "album_type": "Full-length"},
            {"band_name": "Slayer", "track_name": None,
             "album_name": None, "album_type": "EP"},
        ]
        bands_data = [
            {"name": "Iron Maiden", "genre": "Heavy Metal", "label": "EMI"},
            {"name": "Opeth", "genre": None, "label": None},
        ]

        mock_load.side_effect = [
            _make_hf_dataset(tracks_data),
            _make_hf_dataset(bands_data),
        ]

        ner = EncyclopediaMetallvmNER.__new__(EncyclopediaMetallvmNER)
        ner.__init__.__wrapped__ if hasattr(ner.__init__, '__wrapped__') else None
        # Initialize manually to avoid triggering train() in __init__
        from ahocorasick_ner import AhocorasickNER
        AhocorasickNER.__init__(ner)
        ner.load_huggingface()
        ner.fit()

        tags = list(ner.tag("I love Metallica and Iron Maiden", min_word_len=3))
        words = [t["word"] for t in tags]
        self.assertIn("Metallica", words)
        self.assertIn("Iron Maiden", words)

    @patch("ahocorasick_ner.datasets.load_dataset")
    def test_init_with_path_loads(self, mock_load: MagicMock) -> None:
        # Create a real saved automaton
        from ahocorasick_ner import AhocorasickNER
        base = AhocorasickNER()
        base.add_word("test", "hello")
        base.fit()
        with tempfile.NamedTemporaryFile(suffix=".ahocorasick", delete=False) as f:
            path = f.name
        try:
            base.save(path)
            ner = EncyclopediaMetallvmNER(path=path)
            self.assertTrue(ner._fitted)
            mock_load.assert_not_called()
        finally:
            os.unlink(path)

    @patch("ahocorasick_ner.datasets.load_dataset")
    def test_train_saves_if_path(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _make_hf_dataset([])

        from ahocorasick_ner import AhocorasickNER
        ner = EncyclopediaMetallvmNER.__new__(EncyclopediaMetallvmNER)
        AhocorasickNER.__init__(ner)

        with tempfile.NamedTemporaryFile(suffix=".ahocorasick", delete=False) as f:
            path = f.name
        try:
            ner.train(path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    @patch("ahocorasick_ner.datasets.load_dataset")
    def test_train_no_save(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _make_hf_dataset([])
        from ahocorasick_ner import AhocorasickNER
        ner = EncyclopediaMetallvmNER.__new__(EncyclopediaMetallvmNER)
        AhocorasickNER.__init__(ner)
        ner.train(None)  # no path = no save


class TestMusicNER(unittest.TestCase):
    """Tests for the Music NER multi-genre loader."""

    @patch("ahocorasick_ner.datasets.load_dataset")
    def test_load_huggingface_all_genres(self, mock_load: MagicMock) -> None:
        # 7 calls: metallvm tracks, metallvm bands, jazz, prog, classical, trance
        trance = [{"ARTIST(S)": "Armin van Buuren", "TRACK": "Blah Blah Blah", "STYLE": "Trance"}]
        classical = [{"name": "Beethoven"}]
        prog = [{"artist": "Yes", "genre": "Progressive Rock"}]
        jazz = [{"artist": "Miles Davis", "genre": "Bebop"}]
        metal_tracks = [{"band_name": "Metallica", "track_name": "One",
                         "album_name": "AJFA", "album_type": "Full-length"}]
        metal_bands = [{"name": "Metallica", "genre": "Thrash Metal", "label": "Elektra"}]

        mock_load.side_effect = [
            _make_hf_dataset(metal_tracks),
            _make_hf_dataset(metal_bands),
            _make_hf_dataset(jazz),
            _make_hf_dataset(prog),
            _make_hf_dataset(classical),
            _make_hf_dataset(trance),
        ]

        from ahocorasick_ner import AhocorasickNER
        ner = MusicNER.__new__(MusicNER)
        AhocorasickNER.__init__(ner)
        ner.load_huggingface()
        ner.fit()

        tags = list(ner.tag("I love Metallica and Miles Davis and Beethoven", min_word_len=3))
        words = [t["word"] for t in tags]
        self.assertIn("Metallica", words)
        self.assertIn("Miles Davis", words)
        self.assertIn("Beethoven", words)

    @patch("ahocorasick_ner.datasets.load_dataset")
    def test_init_with_path(self, mock_load: MagicMock) -> None:
        from ahocorasick_ner import AhocorasickNER
        base = AhocorasickNER()
        base.add_word("test", "hello")
        base.fit()
        with tempfile.NamedTemporaryFile(suffix=".ahocorasick", delete=False) as f:
            path = f.name
        try:
            base.save(path)
            ner = MusicNER(path=path)
            self.assertTrue(ner._fitted)
            mock_load.assert_not_called()
        finally:
            os.unlink(path)

    @patch("ahocorasick_ner.datasets.load_dataset")
    def test_train_saves(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _make_hf_dataset([])
        from ahocorasick_ner import AhocorasickNER
        ner = MusicNER.__new__(MusicNER)
        AhocorasickNER.__init__(ner)
        with tempfile.NamedTemporaryFile(suffix=".ahocorasick", delete=False) as f:
            path = f.name
        try:
            ner.train(path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)


class TestImdbNER(unittest.TestCase):
    """Tests for the IMDB NER loader."""

    @patch("ahocorasick_ner.datasets.load_dataset")
    def test_load_huggingface(self, mock_load: MagicMock) -> None:
        actors = [{"name": "Robert De Niro"}, {"name": None}]
        directors = [{"name": "Martin Scorsese"}]
        producers = [{"name": "Steven Spielberg"}]
        writers = [{"name": "Aaron Sorkin"}]
        composers = [{"name": "Hans Zimmer"}]

        mock_load.side_effect = [
            _make_hf_dataset(actors),
            _make_hf_dataset(directors),
            _make_hf_dataset(producers),
            _make_hf_dataset(writers),
            _make_hf_dataset(composers),
        ]

        from ahocorasick_ner import AhocorasickNER
        ner = ImdbNER.__new__(ImdbNER)
        AhocorasickNER.__init__(ner)
        ner.load_huggingface()
        ner.fit()

        tags = list(ner.tag("Robert De Niro directed by Martin Scorsese with music by Hans Zimmer",
                            min_word_len=3))
        words = [t["word"] for t in tags]
        self.assertIn("Robert De Niro", words)
        self.assertIn("Martin Scorsese", words)
        self.assertIn("Hans Zimmer", words)

    @patch("ahocorasick_ner.datasets.load_dataset")
    def test_init_with_path(self, mock_load: MagicMock) -> None:
        from ahocorasick_ner import AhocorasickNER
        base = AhocorasickNER()
        base.add_word("test", "hello")
        base.fit()
        with tempfile.NamedTemporaryFile(suffix=".ahocorasick", delete=False) as f:
            path = f.name
        try:
            base.save(path)
            ner = ImdbNER(path=path)
            self.assertTrue(ner._fitted)
            mock_load.assert_not_called()
        finally:
            os.unlink(path)

    @patch("ahocorasick_ner.datasets.load_dataset")
    def test_init_no_path_calls_train(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _make_hf_dataset([])
        ner = ImdbNER()
        # Should have called load_dataset (5 datasets)
        self.assertTrue(mock_load.called)


if __name__ == "__main__":
    unittest.main()
