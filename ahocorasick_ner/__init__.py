import ahocorasick
import pickle
import re
from typing import Dict, Iterable, List, Tuple, Set, Union


class AhocorasickNER:
    """
    A fast, dictionary-based Named Entity Recognition (NER) system using the Aho-Corasick algorithm.
    It supports simultaneous matching of multiple entities with word boundary awareness.
    """

    def __init__(self, case_sensitive: bool = False):
        """
        Initializes the NER system.

        Args:
            case_sensitive: bool, whether entity matching should be case-sensitive.
        """
        self.automaton: ahocorasick.Automaton = ahocorasick.Automaton()
        self.case_sensitive: bool = case_sensitive
        self._fitted: bool = False

    def save(self, path: str) -> None:
        """
        Saves the underlying Aho-Corasick automaton to a file.

        Args:
            path: str, destination file path.
        """
        self.automaton.save(path, pickle.dumps)

    def load(self, path: str) -> None:
        """
        Loads an Aho-Corasick automaton from a file.

        Args:
            path: str, source file path.
        """
        self.automaton = ahocorasick.load(path, pickle.loads)
        self._fitted = True

    def add_word(self, label: str, example: str) -> None:
        """
        Adds a labeled entity example to the automaton.

        Args:
            label: str, the entity label (e.g., 'artist_name').
            example: str, the text to be recognized as this entity.
        """
        key = example if self.case_sensitive else example.lower()
        self.automaton.add_word(key, (label, key))
        self._fitted = False

    def fit(self) -> None:
        """
        Finalizes the Aho-Corasick automaton. Must be called before tagging if new words were added.
        """
        if not self._fitted:
            self.automaton.make_automaton()
        self._fitted = True

    def tag(self, haystack: str, min_word_len: int = 5) -> Iterable[Dict[str, Union[int, str]]]:
        """
        Searches for registered entities in the input text.
        Implements greedy longest-match-first strategy and respects word boundaries.

        Args:
            haystack: str, the text to search.
            min_word_len: int, minimum length of a match to be considered valid.

        Yields:
            Dict[str, Union[int, str]]: Dictionary with 'start', 'end', 'word', and 'label'.
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
