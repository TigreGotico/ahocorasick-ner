"""ONNX-based Aho-Corasick NER inference backend.

Exports the FSM as an ONNX model (Loop + Gather ops) for portable,
zero-Python inference via onnxruntime. Post-processing (word boundaries,
greedy resolution) runs in numpy.
"""
from collections import deque
from typing import Dict, Iterable, List, Set, Tuple, Union

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER, _is_word_char


def _fill_goto_table(goto: np.ndarray, failure: np.ndarray) -> np.ndarray:
    """Pre-fills the goto table by following failure links.

    After filling, ``goto[s][c]`` always gives the correct next state
    without needing failure links at runtime.

    Args:
        goto: Original transition table ``[num_states, alphabet_size]``.
        failure: Failure links ``[num_states]``.

    Returns:
        Filled goto table with same shape.
    """
    filled = goto.copy()
    alpha = goto.shape[1]
    queue: deque[int] = deque()

    for c in range(alpha):
        if filled[0, c] != 0:
            queue.append(int(filled[0, c]))

    while queue:
        s = queue.popleft()
        for c in range(alpha):
            t = int(filled[s, c])
            if t != 0:
                queue.append(t)
            else:
                filled[s, c] = filled[int(failure[s]), c]

    return filled


def export_onnx(
    goto: np.ndarray,
    failure: np.ndarray,
    output: np.ndarray,
    output_counts: np.ndarray,
    char_to_id: np.ndarray,
    path: str,
) -> None:
    """Exports the Aho-Corasick FSM tables as an ONNX model.

    The ONNX graph takes a 1-D int64 tensor of character IDs as input and
    returns per-position state IDs. Post-processing (output lookup, boundary
    check, greedy resolution) is done outside the ONNX graph.

    Args:
        goto: Transition table ``[num_states, alphabet_size]``.
        failure: Failure links ``[num_states]``.
        output: Output table ``[num_states, max_out, 2]``.
        output_counts: Output counts ``[num_states]``.
        char_to_id: Character ordinal to char_id mapping.
        path: Destination ``.onnx`` file path.
    """
    filled_goto = _fill_goto_table(goto, failure)
    alpha = filled_goto.shape[1]
    goto_flat = filled_goto.reshape(-1).astype(np.int64)

    # --- Loop body subgraph ---
    # ONNX Loop body inputs: (iteration_num, cond_in, ...carried_states)
    # No scan inputs — we access char_ids from outer scope via the graph input name.
    # Body outputs: (cond_out, ...carried_states, ...scan_outputs)

    iter_in = helper.make_tensor_value_info("iter_num", TensorProto.INT64, [])
    cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    state_prev = helper.make_tensor_value_info("state_prev", TensorProto.INT64, [])

    cond_out_vi = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    state_out_vi = helper.make_tensor_value_info("state_out", TensorProto.INT64, [])
    state_log_vi = helper.make_tensor_value_info("state_log", TensorProto.INT64, [])

    alpha_const = numpy_helper.from_array(np.int64(alpha), name="alpha_const")
    true_const = numpy_helper.from_array(np.array(True), name="true_const")

    body_nodes = [
        # char_id = char_ids[iter_num]  (char_ids from outer scope)
        helper.make_node("Gather", ["char_ids", "iter_num"], ["char_id"], axis=0),
        # goto_idx = state_prev * alpha + char_id
        helper.make_node("Mul", ["state_prev", "alpha_const"], ["sa"]),
        helper.make_node("Add", ["sa", "char_id"], ["goto_idx"]),
        # next_state = goto_flat[goto_idx]
        helper.make_node("Gather", ["goto_flat", "goto_idx"], ["state_out"], axis=0),
        helper.make_node("Identity", ["state_out"], ["state_log"]),
        helper.make_node("Identity", ["true_const"], ["cond_out"]),
    ]

    body_graph = helper.make_graph(
        body_nodes,
        "loop_body",
        [iter_in, cond_in, state_prev],
        [cond_out_vi, state_out_vi, state_log_vi],
        initializer=[alpha_const, true_const],
    )

    # --- Main graph ---
    char_ids_input = helper.make_tensor_value_info("char_ids", TensorProto.INT64, [None])
    states_output = helper.make_tensor_value_info("states", TensorProto.INT64, [None])

    goto_flat_init = numpy_helper.from_array(goto_flat, name="goto_flat")
    zero_scalar = numpy_helper.from_array(np.int64(0), name="zero_scalar")
    zero_1d = numpy_helper.from_array(np.array([0], dtype=np.int64), name="zero_1d")
    true_init = numpy_helper.from_array(np.array(True), name="true_init")

    main_nodes = [
        # seq_len = shape(char_ids)[0]
        helper.make_node("Shape", ["char_ids"], ["shape_arr"]),
        helper.make_node("Gather", ["shape_arr", "zero_1d"], ["seq_len_1d"], axis=0),
        helper.make_node("Squeeze", ["seq_len_1d"], ["seq_len"]),
        # Loop: carried state = current FSM state; scan output = state at each step
        helper.make_node(
            "Loop",
            ["seq_len", "true_init", "zero_scalar"],
            ["_state_final", "states"],
            body=body_graph,
        ),
    ]

    graph = helper.make_graph(
        main_nodes,
        "aho_corasick_fsm",
        [char_ids_input],
        [states_output],
        initializer=[goto_flat_init, zero_scalar, zero_1d, true_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


class OnnxAhocorasickNER:
    """Dictionary-based NER using an ONNX-exported Aho-Corasick automaton.

    API-compatible with :class:`AhocorasickNER`. Build with :meth:`add_word` /
    :meth:`fit`, export to ONNX with :meth:`save`, run inference with :meth:`tag`.
    """

    def __init__(self, case_sensitive: bool = False) -> None:
        """Initializes the NER system.

        Args:
            case_sensitive: Whether entity matching should be case-sensitive.
        """
        self.case_sensitive: bool = case_sensitive
        self._inner: NumpyAhocorasickNER = NumpyAhocorasickNER(case_sensitive=case_sensitive)
        self._session: ort.InferenceSession | None = None
        self._fitted: bool = False

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
        self._inner.add_word(label, example)
        self._fitted = False

    def fit(self) -> None:
        """Finalizes the automaton and prepares the ONNX session."""
        if not self._fitted:
            self._inner.fit()
            self._output = self._inner._output
            self._output_counts = self._inner._output_counts
            self._char_to_id = self._inner._char_to_id
            self._label_strings = self._inner._label_strings
            import tempfile
            import os
            tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                export_onnx(
                    self._inner._goto,
                    self._inner._failure,
                    self._output,
                    self._output_counts,
                    self._char_to_id,
                    tmp_path,
                )
                self._session = ort.InferenceSession(tmp_path)
            finally:
                os.unlink(tmp_path)
        self._fitted = True

    def save(self, path: str) -> None:
        """Saves the automaton as ONNX model + numpy side tables.

        Creates two files: ``<path>.onnx`` and ``<path>.npz``.

        Args:
            path: Base file path (without extension).
        """
        if not self._fitted:
            self.fit()
        onnx_path = path if path.endswith(".onnx") else path + ".onnx"
        export_onnx(
            self._inner._goto,
            self._inner._failure,
            self._output,
            self._output_counts,
            self._char_to_id,
            onnx_path,
        )
        npz_path = path.replace(".onnx", "") + ".npz"
        # Convert label_strings to fixed-length string array for safe serialization
        label_strings = self._label_strings
        if label_strings.dtype == object:
            # Find max length
            max_len = max(len(s) for s in label_strings) if len(label_strings) > 0 else 0
            label_strings = np.array(label_strings, dtype=f'U{max_len}')

        np.savez(
            npz_path,
            goto=self._inner._goto,
            failure=self._inner._failure,
            output=self._output,
            output_counts=self._output_counts,
            char_to_id=self._char_to_id,
            label_strings=label_strings,
            case_sensitive=np.array([self.case_sensitive]),
        )

    def load(self, path: str) -> None:
        """Loads an ONNX model + side tables.

        Args:
            path: Base file path. Expects ``<path>.onnx`` and ``<path>.npz``.
        """
        base = path.replace(".onnx", "").replace(".npz", "")
        self._session = ort.InferenceSession(base + ".onnx")
        # Try loading without pickle first (secure)
        try:
            data = np.load(base + ".npz", allow_pickle=False)
        except ValueError:
            # Fall back to allow_pickle=True for backward compatibility with old files
            data = np.load(base + ".npz", allow_pickle=True)
        self._output = data["output"]
        self._output_counts = data["output_counts"]
        self._char_to_id = data["char_to_id"]
        self._label_strings = data["label_strings"]
        if "case_sensitive" in data:
            self.case_sensitive = bool(data["case_sensitive"][0])
        # Restore _inner's transition tables so subsequent save() exports correctly
        if "goto" in data:
            self._inner._goto = data["goto"]
            self._inner._failure = data["failure"]
        self._fitted = True

    def _char_id(self, ch: str) -> int:
        """Maps a character to its integer ID. Returns 0 for unknown chars."""
        o = ord(ch)
        if o < len(self._char_to_id):
            return int(self._char_to_id[o])
        return 0

    def tag(self, haystack: str, min_word_len: int = 5) -> Iterable[Dict[str, Union[int, str]]]:
        """Searches for registered entities in the input text.

        FSM traversal runs in onnxruntime; post-processing in numpy.

        Args:
            haystack: Text to search.
            min_word_len: Minimum match length to be considered valid.

        Yields:
            Dict with 'start', 'end', 'word', and 'label' keys.
        """
        if not self._fitted:
            self.fit()

        processed = haystack if self.case_sensitive else haystack.lower()
        char_ids = np.array([self._char_id(ch) for ch in processed], dtype=np.int64)

        assert self._session is not None
        states = self._session.run(None, {"char_ids": char_ids})[0]

        output = self._output
        output_counts = self._output_counts
        label_strings = self._label_strings

        matches: List[Tuple[int, int, str, str]] = []
        for idx in range(len(processed)):
            s = int(states[idx])
            n_out = int(output_counts[s])
            for j in range(n_out):
                lid = int(output[s, j, 0])
                mlen = int(output[s, j, 1])
                if mlen < min_word_len:
                    continue
                start = idx - mlen + 1
                end = idx
                before = processed[start - 1] if start > 0 else " "
                after = processed[end + 1] if end + 1 < len(processed) else " "
                if _is_word_char(before) or _is_word_char(after):
                    continue
                label = str(label_strings[lid])
                matches.append((start, end, processed[start:end + 1], label))

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
