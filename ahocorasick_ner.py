import ahocorasick
import re
import pickle
import os
from typing import Dict, Iterable, List, Tuple, Set

try:
    from datasets import load_dataset
except ImportError as e:
    # only used in demo classes, not a hard requirement
    def load_dataset(*args, **kwargs):
        raise e

class AhocorasickNER:
    """
    A simple Named Entity Recognition system using the Aho-Corasick algorithm.
    Supports matching pre-defined entities in a given string with word boundary filtering.
    """

    def __init__(self, case_sensitive: bool = False):
        """
        Initialize the NER system.

        Args:
            case_sensitive (bool): Whether matching should be case-sensitive. Defaults to False.
        """
        self.automaton = ahocorasick.Automaton()
        self.case_sensitive = case_sensitive
        self._fitted = False

    def save(self, path: str):
        self.automaton.save(path, pickle.dumps)

    def load(self, path: str):
        self.automaton = ahocorasick.load(path, pickle.loads)

    def add_word(self, label: str, example: str) -> None:
        """
        Add a labeled word or phrase to the automaton.

        Args:
            label (str): The label to associate with the word (e.g., 'artist_name').
            example (str): The word or phrase to match.
        """
        key = example if self.case_sensitive else example.lower()
        self.automaton.add_word(key, (label, key))
        self._fitted = False

    def fit(self) -> None:
        """
        Finalize the automaton. This must be called after all words are added.
        """
        if not self._fitted:
            self.automaton.make_automaton()
        self._fitted = True

    def tag(self, haystack: str, min_word_len: int = 5) -> Iterable[Dict[str, str]]:
        """
        Search for labeled entities in the given string.

        Args:
            haystack (str): The input string to search.
            min_word_len (int): Minimum word length to consider a match. Defaults to 5.

        Yields:
            Dict[str, str]: A dictionary with keys 'start', 'end', 'word', and 'label'.
        """
        if not self._fitted:
            self.fit()

        processed_haystack = haystack if self.case_sensitive else haystack.lower()
        matches: List[Tuple[int, int, str, str]] = []

        for idx, (label, word) in self.automaton.iter(processed_haystack):
            if len(word) < min_word_len:
                continue

            start = idx - len(word) + 1
            end = idx

            # Respect word boundaries
            before = processed_haystack[start - 1] if start > 0 else ' '
            after = processed_haystack[end + 1] if end + 1 < len(processed_haystack) else ' '
            if re.match(r'\w', before) or re.match(r'\w', after):
                continue  # skip partial word matches

            matches.append((start, end, word, label))

        # Sort by descending length, then by start position
        matches.sort(key=lambda x: (-(x[1] - x[0] + 1), x[0]))

        selected: List[Tuple[int, int, str, str]] = []
        used: Set[int] = set()

        for start, end, word, label in matches:
            if all(i not in used for i in range(start, end + 1)):
                selected.append((start, end, word, label))
                used.update(range(start, end + 1))

        for start, end, word, label in sorted(selected, key=lambda x: x[0]):
            yield {
                "start": start,
                "end": end,
                "word": haystack[start:end + 1],
                "label": label
            }


### just for demonstration
class EncyclopediaMetallvmNER(AhocorasickNER):
    def __init__(self, path: str | None=None ,case_sensitive: bool = False):
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.train(path)

    def train(self, path: str | None=None ):
        self.load_huggingface()
        if path:
            self.save(path)

    def load_huggingface(self):
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
    def __init__(self, path: str | None=None ,
                 case_sensitive: bool = False):
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.train(path)

    def train(self, path: str | None=None ):
        self.load_huggingface()
        if path:
            self.save(path)

    def load_huggingface(self):
        self.load_metallvm()
        self.load_jazz()
        self.load_prog()
        self.load_classical()
        self.load_trance()

    def load_trance(self):
        dataset_name = "Jarbas/trance_tracks"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("ARTIST(S)"):
                self.add_word("artist_name", entry["ARTIST(S)"])
            if entry.get("TRACK"):
                self.add_word("track_name", entry["TRACK"])
            if entry.get("STYLE"):
                self.add_word("music_genre", entry["STYLE"])

    def load_classical(self):
        dataset_name = "Jarbas/classic-composers"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("artist_name", entry["name"])

    def load_prog(self):
        dataset_name = "Jarbas/prog-archives"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("artist"):
                self.add_word("artist_name", entry["artist"])
            if entry.get("genre"):
                self.add_word("music_genre", entry["genre"])

    def load_jazz(self):
        dataset_name = "Jarbas/jazz-music-archives"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("artist"):
                self.add_word("artist_name", entry["artist"])
            if entry.get("genre"):
                self.add_word("music_genre", entry["genre"])

    def load_metallvm(self):
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
    def __init__(self, path: str | None=None ,case_sensitive: bool = False):
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.train(path)

    def train(self, path: str | None=None ):
        self.load_huggingface()
        if path:
            self.save(path)

    def load_huggingface(self):
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

    #e = MusicNER("media_net.ahocorasick")
    #e = ImdbNER("imdb.ahocorasick")
    e = EncyclopediaMetallvmNER("metallvm.ahocorasick")

    s = time.monotonic()
    for entity in e.tag("I fucking love black metal"):
        print(entity)
    print(time.monotonic() - s)