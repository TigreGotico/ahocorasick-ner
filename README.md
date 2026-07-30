[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/ahocorasick-ner)

# AhocorasickNER

AhocorasickNER is a Named Entity Recognition (NER) tool based on the [Aho-Corasick algorithm](https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm). It matches text against a list of known words and phrases you define. Use it for rule-based entity extraction with pre-defined vocabularies.

---

## Features

- Multi-pattern string matching with the Aho-Corasick algorithm
- Word-boundary-aware matching with greedy longest-match
- Case-sensitive or case-insensitive modes
- Three inference backends: pyahocorasick (C), pure NumPy, and ONNX
- OpenVoiceOS plugin integration

---

## Installation

```bash
uv pip install ahocorasick-ner                    # core (pyahocorasick backend)
uv pip install ahocorasick-ner[numpy]             # + pure numpy backend
uv pip install ahocorasick-ner[onnx]              # + ONNX export/inference
uv pip install ahocorasick-ner[datasets]          # + HuggingFace dataset loaders
```

---

## Quick Start

```python
from ahocorasick_ner import AhocorasickNER

ner = AhocorasickNER()
ner.add_word("city", "New York")
ner.add_word("city", "London")
ner.add_word("country", "Japan")
ner.fit()

for entity in ner.tag("I flew from New York to London, then on to Japan."):
    print(entity)
# {'start': 12, 'end': 19, 'word': 'New York', 'label': 'city'}
# {'start': 24, 'end': 29, 'word': 'London', 'label': 'city'}
# {'start': 43, 'end': 47, 'word': 'Japan', 'label': 'country'}
```

---

## Backends

All three backends share the same `add_word` / `fit` / `tag` / `save` / `load` API.

| Backend | Import | Dependency | Persistence | Use case |
|---------|--------|-----------|-------------|----------|
| pyahocorasick | `from ahocorasick_ner import AhocorasickNER` | `pyahocorasick` (C ext) | `.ahocorasick` (pickle) | Fastest, default |
| numpy | `from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER` | `numpy` | `.npz` | No C deps, portable |
| ONNX | `from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER` | `onnx` + `onnxruntime` | `.onnx` + `.npz` | Edge/WASM deployment |

```python
# Numpy backend: no C extensions at inference
from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER
ner = NumpyAhocorasickNER()
ner.add_word("city", "Tokyo")
ner.fit()
ner.save("model.npz")

# ONNX backend: portable to any onnxruntime deployment
from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER
ner = OnnxAhocorasickNER()
ner.add_word("city", "Tokyo")
ner.fit()
ner.save("model")  # creates model.onnx + model.npz
```

See [`examples/`](examples/) for complete working examples and a benchmark script.

---

## Benchmarks

With 100k+ known phrases, this tool tags documents in milliseconds because the Aho-Corasick FSM matches all patterns in a single pass. Run the benchmark yourself:

```bash
uv pip install ahocorasick-ner[numpy,onnx]
uv run python examples/benchmark.py
```

---

## Limitations

- Greedy longest-match only, so no nested or overlapping entities
- No fuzzy matching (typos or misspellings will not match)
- All entities must be known beforehand

---

## Testing

Run all tests, including the NumPy and ONNX backend tests.

```bash
# Install with test dependencies
uv pip install -e ".[test]"

# Run tests
uv run pytest test/unittests -v

# Run with coverage
uv run pytest test/unittests --cov=ahocorasick_ner --cov-report=term-missing
```

Tests require `numpy`, `onnx`, and `onnxruntime`. These packages are included in the `test` extra.

---

## Related Projects

- [OpenVoiceOS](https://github.com/OpenVoiceOS): the voice assistant platform this library's `opm` extra plugs into (see `ahocorasick_ner/opm.py`)
- [simple_NER](https://github.com/TigreGotico/simple_NER): another TigreGotico NER library, referenced in the [dataset reference docs](docs/DATASET_REFERENCE.md)

---

## License

Apache 2.0. Free for commercial and non-commercial use.

---

## Acknowledgements

- [pyahocorasick](https://github.com/WojciechMula/pyahocorasick): C-based Aho-Corasick implementation
- [Hugging Face Datasets](https://huggingface.co/docs/datasets): domain-specific corpora
