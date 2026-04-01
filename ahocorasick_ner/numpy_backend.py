"""Numpy-based Aho-Corasick NER inference backend.

Drop-in replacement for ``AhocorasickNER`` that uses pure numpy arrays
instead of the ``pyahocorasick`` C extension. No C dependencies at inference time.
"""
from typing import Dict, Iterable, List, Set, Tuple, Union

import numpy as np

from ahocorasick_ner.automaton import AhocorasickAutomaton


def _is_word_char(ch: str) -> bool:
    """Returns True if *ch* matches ``\\w`` (alphanumeric or underscore)."""
    return ch.isalnum() or ch == "_"


class NumpyAhocorasickNER:
    """Dictionary-based NER using a pure-numpy Aho-Corasick automaton.

    API-compatible with :class:`AhocorasickNER`. Build the automaton with
    :meth:`add_word` / :meth:`fit`, then tag text with :meth:`tag`.
    Persistence via ``.npz`` files (:meth:`save` / :meth:`load`).
    """

    def __init__(self, case_sensitive: bool = False) -> None:
        """Initializes the NER system.

        Args:
            case_sensitive: Whether entity matching should be case-sensitive.
        """
        self.case_sensitive: bool = case_sensitive
        self._automaton: AhocorasickAutomaton = AhocorasickAutomaton()
        self._fitted: bool = False

        # Populated by fit() or load()
        self._goto: np.ndarray = np.empty(0, dtype=np.int32)
        self._failure: np.ndarray = np.empty(0, dtype=np.int32)
        self._output: np.ndarray = np.empty(0, dtype=np.int32)
        self._output_counts: np.ndarray = np.empty(0, dtype=np.int32)
        self._char_to_id: np.ndarray = np.empty(0, dtype=np.int32)
        self._label_strings: np.ndarray = np.empty(0, dtype=np.str_)

    def add_word(self, label: str, example: str) -> None:
        """Adds a labeled entity example to the automaton.

        Args:
            label: Entity label (e.g. 'artist_name').
            example: Text to be recognized as this entity.
        """
        key = example if self.case_sensitive else example.lower()
        self._automaton.add_word(label, key)
        self._fitted = False

    def fit(self) -> None:
        """Finalizes the automaton. Must be called before tagging."""
        if not self._fitted:
            self._automaton.build()
            tables = self._automaton.to_tables()
            self._goto = tables["goto"]
            self._failure = tables["failure"]
            self._output = tables["output"]
            self._output_counts = tables["output_counts"]
            self._char_to_id = tables["char_to_id"]
            self._label_strings = tables["label_strings"]
        self._fitted = True

    def save(self, path: str) -> None:
        """Saves the automaton tables to a ``.npz`` file.

        Args:
            path: Destination file path.
        """
        if not self._fitted:
            self.fit()
        # Convert label_strings to fixed-length string array for safe serialization
        label_strings = self._label_strings
        if label_strings.dtype == object:
            # Find max length
            max_len = max(len(s) for s in label_strings) if len(label_strings) > 0 else 0
            label_strings = np.array(label_strings, dtype=f'U{max_len}')

        np.savez(
            path,
            goto=self._goto,
            failure=self._failure,
            output=self._output,
            output_counts=self._output_counts,
            char_to_id=self._char_to_id,
            label_strings=label_strings,
            case_sensitive=np.array([self.case_sensitive]),
        )

    def load(self, path: str) -> None:
        """Loads automaton tables from a ``.npz`` file.

        Args:
            path: Source file path (with or without ``.npz`` extension).
        """
        file_path = path if path.endswith(".npz") else path + ".npz"
        # Try loading without pickle first (secure)
        try:
            data = np.load(file_path, allow_pickle=False)
        except ValueError:
            # Fall back to allow_pickle=True for backward compatibility with old files
            data = np.load(file_path, allow_pickle=True)
        self._goto = data["goto"]
        self._failure = data["failure"]
        self._output = data["output"]
        self._output_counts = data["output_counts"]
        self._char_to_id = data["char_to_id"]
        self._label_strings = data["label_strings"]
        if "case_sensitive" in data:
            self.case_sensitive = bool(data["case_sensitive"][0])
        self._fitted = True

    def _char_id(self, ch: str) -> int:
        """Maps a character to its integer ID. Returns 0 for unknown chars."""
        o = ord(ch)
        if o < len(self._char_to_id):
            return int(self._char_to_id[o])
        return 0

    def tag(self, haystack: str, min_word_len: int = 5) -> Iterable[Dict[str, Union[int, str]]]:
        """Searches for registered entities in the input text.

        Implements greedy longest-match-first strategy with word boundary awareness.

        Args:
            haystack: Text to search.
            min_word_len: Minimum match length to be considered valid.

        Yields:
            Dict with 'start', 'end', 'word', and 'label' keys.
        """
        if not self._fitted:
            self.fit()

        processed = haystack if self.case_sensitive else haystack.lower()
        goto = self._goto
        failure = self._failure
        output = self._output
        output_counts = self._output_counts
        label_strings = self._label_strings

        # FSM traversal — sequential per character
        matches: List[Tuple[int, int, str, str]] = []
        state = 0
        for idx, ch in enumerate(processed):
            cid = self._char_id(ch)
            # Follow failure links until we find a transition or reach root
            while state != 0 and goto[state, cid] == 0:
                state = int(failure[state])
            state = int(goto[state, cid])

            # Collect outputs at this state
            n_out = int(output_counts[state])
            for j in range(n_out):
                lid = int(output[state, j, 0])
                mlen = int(output[state, j, 1])
                if mlen < min_word_len:
                    continue
                start = idx - mlen + 1
                end = idx
                # Word boundary check
                before = processed[start - 1] if start > 0 else " "
                after = processed[end + 1] if end + 1 < len(processed) else " "
                if _is_word_char(before) or _is_word_char(after):
                    continue
                label = str(label_strings[lid])
                word_key = processed[start:end + 1]
                matches.append((start, end, word_key, label))

        # Greedy longest-match conflict resolution
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
                "label": label,
            }
