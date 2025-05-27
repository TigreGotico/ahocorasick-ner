import ahocorasick
import pickle
import re
from typing import Dict, Iterable, List, Tuple, Set


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
