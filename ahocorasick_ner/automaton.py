"""Pure-Python Aho-Corasick automaton builder with numpy table export.

Builds a trie with failure links (BFS) and exports the FSM as flat numpy arrays
for use by the numpy and ONNX inference backends.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class _TrieNode:
    """Internal trie node used during automaton construction."""
    goto: Dict[str, int] = field(default_factory=dict)
    failure: int = 0
    outputs: List[Tuple[int, int]] = field(default_factory=list)  # (label_id, match_length)


class AhocorasickAutomaton:
    """Pure-Python Aho-Corasick automaton builder.

    Constructs a trie with failure links and exports the FSM as numpy arrays.
    Character-level matching (not byte-level) for compatibility with pyahocorasick.

    The automaton maps characters to integer IDs via a vocabulary built during
    ``add_word``. Unknown characters at inference time map to ID 0 (no transition).
    """

    def __init__(self) -> None:
        self._nodes: List[_TrieNode] = [_TrieNode()]  # root = state 0
        self._labels: Dict[str, int] = {}  # label string -> label_id
        self._labels_rev: List[str] = []  # label_id -> label string
        self._char_to_id: Dict[str, int] = {}  # char -> int (1-indexed; 0 = unknown)
        self._alphabet_size: int = 1  # next char id to assign
        self._built: bool = False

    def _get_char_id(self, ch: str) -> int:
        """Returns the integer ID for a character, assigning a new one if needed."""
        cid = self._char_to_id.get(ch)
        if cid is None:
            cid = self._alphabet_size
            self._char_to_id[ch] = cid
            self._alphabet_size += 1
        return cid

    def _get_label_id(self, label: str) -> int:
        """Returns the integer ID for a label, assigning a new one if needed."""
        lid = self._labels.get(label)
        if lid is None:
            lid = len(self._labels_rev)
            self._labels[label] = lid
            self._labels_rev.append(label)
        return lid

    def add_word(self, label: str, word: str) -> None:
        """Adds a labeled word to the automaton.

        Args:
            label: Entity label (e.g. 'artist_name').
            word: The text pattern to match (already normalized by caller).
        """
        self._built = False
        lid = self._get_label_id(label)
        state = 0
        for ch in word:
            cid = self._get_char_id(ch)
            nxt = self._nodes[state].goto.get(ch)
            if nxt is None:
                nxt = len(self._nodes)
                self._nodes.append(_TrieNode())
                self._nodes[state].goto[ch] = nxt
            state = nxt
        self._nodes[state].outputs.append((lid, len(word)))

    def build(self) -> None:
        """Constructs failure links via BFS (standard Aho-Corasick construction)."""
        root = self._nodes[0]
        queue: deque[int] = deque()

        # Depth-1 nodes: failure -> root
        for ch, s in root.goto.items():
            self._nodes[s].failure = 0
            queue.append(s)

        # BFS
        while queue:
            u = queue.popleft()
            u_node = self._nodes[u]
            for ch, v in u_node.goto.items():
                queue.append(v)
                # Walk failure links to find failure state for v
                f = u_node.failure
                while f != 0 and ch not in self._nodes[f].goto:
                    f = self._nodes[f].failure
                self._nodes[v].failure = self._nodes[f].goto.get(ch, 0)
                if self._nodes[v].failure == v:
                    self._nodes[v].failure = 0
                # Merge outputs from failure state
                self._nodes[v].outputs = self._nodes[v].outputs + self._nodes[self._nodes[v].failure].outputs

        self._built = True

    def to_tables(self) -> Dict[str, np.ndarray]:
        """Exports the automaton as flat numpy arrays.

        Returns:
            Dictionary with keys:
            - ``goto``: int32 array ``[num_states, alphabet_size]`` — transition table
            - ``failure``: int32 array ``[num_states]`` — failure links
            - ``output``: int32 array ``[num_states, max_outputs, 2]`` — (label_id, match_length)
            - ``output_counts``: int32 array ``[num_states]`` — number of outputs per state
            - ``char_to_id``: int32 array ``[max_ord + 1]`` — character ordinal to char_id mapping
            - ``label_strings``: object array of label strings (for decoding label_ids)
        """
        if not self._built:
            self.build()

        num_states = len(self._nodes)
        alpha = self._alphabet_size

        # Goto table
        goto = np.zeros((num_states, alpha), dtype=np.int32)
        for s, node in enumerate(self._nodes):
            for ch, nxt in node.goto.items():
                cid = self._char_to_id[ch]
                goto[s, cid] = nxt

        # Failure table
        failure = np.array([n.failure for n in self._nodes], dtype=np.int32)

        # Output table (padded)
        max_out = max((len(n.outputs) for n in self._nodes), default=0)
        max_out = max(max_out, 1)  # at least 1 to avoid zero-dim
        output = np.zeros((num_states, max_out, 2), dtype=np.int32)
        output_counts = np.zeros(num_states, dtype=np.int32)
        for s, node in enumerate(self._nodes):
            output_counts[s] = len(node.outputs)
            for i, (lid, mlen) in enumerate(node.outputs):
                output[s, i, 0] = lid
                output[s, i, 1] = mlen

        # Char-to-id mapping: ordinal -> char_id
        max_ord = max((ord(ch) for ch in self._char_to_id), default=0)
        char_to_id = np.zeros(max_ord + 2, dtype=np.int32)  # +2 for safety
        for ch, cid in self._char_to_id.items():
            char_to_id[ord(ch)] = cid

        label_strings = np.array(self._labels_rev, dtype=object)

        return {
            "goto": goto,
            "failure": failure,
            "output": output,
            "output_counts": output_counts,
            "char_to_id": char_to_id,
            "label_strings": label_strings,
        }

    @property
    def labels(self) -> List[str]:
        """Returns the list of label strings in label_id order."""
        return list(self._labels_rev)
