import os
from typing import Optional, Union
from ahocorasick_ner import AhocorasickNER

try:
    from datasets import load_dataset
except ImportError:
    # only used in demo classes, not a hard requirement
    def load_dataset(*args, **kwargs):
        raise ImportError("The 'datasets' library is required to use this functionality. "
                          "Install it with 'pip install datasets'.")


class EncyclopediaMetallvmNER(AhocorasickNER):
    """
    NER system pre-loaded with band, track, and album names from Encyclopedia Metallum
    via Hugging Face datasets.
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        """
        Initializes the Encyclopedia Metallum NER.

        Args:
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
        """
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.train(path)

    def train(self, path: Optional[str] = None) -> None:
        """
        Loads data from Hugging Face and optionally saves the automaton.

        Args:
            path: Optional path to save the trained automaton.
        """
        self.load_huggingface()
        if path:
            self.save(path)

    def load_huggingface(self) -> None:
        """
        Loads Metal Archives data from Hugging Face.
        """
        dataset_name = "Jarbas/metal-archives-tracks"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            self.add_word("artist_name", entry["band_name"])
            if entry.get("track_name"):
                self.add_word("track_name", entry["track_name"])
            if entry.get("album_name"):
                self.add_word("album_name", entry["album_name"])
            self.add_word("album_type", entry["album_type"])

        dataset_name = "Jarbas/metal-archives-bands"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            self.add_word("artist_name", entry["name"])
            if entry.get("genre"):
                self.add_word("music_genre", entry["genre"])
            if entry.get("label"):
                self.add_word("record_label", entry["label"])


class MusicNER(AhocorasickNER):
    """
    A comprehensive music NER system combining multiple genres (Metal, Jazz, Prog, Classical, Trance).
    """

    def __init__(self, path: Optional[str] = None,
                 case_sensitive: bool = False):
        """
        Initializes the Music NER.

        Args:
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
        """
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.train(path)

    def train(self, path: Optional[str] = None) -> None:
        """
        Loads data from multiple music datasets and optionally saves the automaton.

        Args:
            path: Optional path to save the trained automaton.
        """
        self.load_huggingface()
        if path:
            self.save(path)

    def load_huggingface(self) -> None:
        """
        Loads all music sub-datasets.
        """
        self.load_metallvm()
        self.load_jazz()
        self.load_prog()
        self.load_classical()
        self.load_trance()

    def load_trance(self) -> None:
        dataset_name = "Jarbas/trance_tracks"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("ARTIST(S)"):
                self.add_word("artist_name", entry["ARTIST(S)"])
            if entry.get("TRACK"):
                self.add_word("track_name", entry["TRACK"])
            if entry.get("STYLE"):
                self.add_word("music_genre", entry["STYLE"])

    def load_classical(self) -> None:
        dataset_name = "Jarbas/classic-composers"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("artist_name", entry["name"])

    def load_prog(self) -> None:
        dataset_name = "Jarbas/prog-archives"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("artist"):
                self.add_word("artist_name", entry["artist"])
            if entry.get("genre"):
                self.add_word("music_genre", entry["genre"])

    def load_jazz(self) -> None:
        dataset_name = "Jarbas/jazz-music-archives"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("artist"):
                self.add_word("artist_name", entry["artist"])
            if entry.get("genre"):
                self.add_word("music_genre", entry["genre"])

    def load_metallvm(self) -> None:
        dataset_name = "Jarbas/metal-archives-tracks"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            self.add_word("artist_name", entry["band_name"])
            if entry.get("track_name"):
                self.add_word("track_name", entry["track_name"])
            if entry.get("album_name"):
                self.add_word("album_name", entry["album_name"])
            self.add_word("album_type", entry["album_type"])

        dataset_name = "Jarbas/metal-archives-bands"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            self.add_word("artist_name", entry["name"])
            if entry.get("genre"):
                self.add_word("music_genre", entry["genre"])
            if entry.get("label"):
                self.add_word("record_label", entry["label"])


class ImdbNER(AhocorasickNER):
    """
    NER system for movie-related entities (actors, directors, producers, etc.) via IMDB data.
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        """
        Initializes the IMDB NER.

        Args:
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
        """
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.train(path)

    def train(self, path: Optional[str] = None) -> None:
        """
        Loads data from Hugging Face and optionally saves the automaton.

        Args:
            path: Optional path to save the trained automaton.
        """
        self.load_huggingface()
        if path:
            self.save(path)

    def load_huggingface(self) -> None:
        """
        Loads various IMDB datasets from Hugging Face.
        """
        dataset_name = "Jarbas/movie_actors"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("movie_actor", entry["name"])

        dataset_name = "Jarbas/movie_directors"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("movie_director", entry["name"])

        dataset_name = "Jarbas/movie_producers"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("movie_producer", entry["name"])

        dataset_name = "Jarbas/movie_writers"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("movie_writer", entry["name"])

        dataset_name = "Jarbas/movie_composers"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("movie_composer", entry["name"])


if __name__ == "__main__":
    import time

    # e = MusicNER("media_net.ahocorasick")
    # e = ImdbNER("imdb.ahocorasick")
    e = EncyclopediaMetallvmNER("metallvm.ahocorasick")

    s = time.monotonic()
    for entity in e.tag("I fucking love black metal"):
        print(entity)
    print(time.monotonic() - s)
