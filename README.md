[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/ahocorasick-ner)

# AhocorasickNER

A fast and simple Named Entity Recognition (NER) tool based on the Aho-Corasick algorithm. This package is ideal for rule-based entity extraction using pre-defined vocabularies, especially when speed and scalability matter.

---

## Features

- Ultra-fast multi-pattern string matching using [Aho-Corasick](https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm)
- Word-boundary-aware matching with greedy longest-match
- Case-sensitive or case-insensitive modes
- **Three inference backends**: pyahocorasick (C), pure numpy, ONNX
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
# {'start': 14, 'end': 21, 'word': 'New York', 'label': 'city'}
# {'start': 26, 'end': 31, 'word': 'London', 'label': 'city'}
# {'start': 44, 'end': 48, 'word': 'Japan', 'label': 'country'}
```

---

## Backends

All three backends share the same `add_word` / `fit` / `tag` / `save` / `load` API.

| Backend | Import | Dependency | Persistence | Use case |
|---------|--------|-----------|-------------|----------|
| **pyahocorasick** | `from ahocorasick_ner import AhocorasickNER` | `pyahocorasick` (C ext) | `.ahocorasick` (pickle) | Fastest; default |
| **numpy** | `from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER` | `numpy` | `.npz` | No C deps; portable |
| **ONNX** | `from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER` | `onnx` + `onnxruntime` | `.onnx` + `.npz` | Edge/WASM deployment |

```python
# Numpy backend — no C extensions at inference
from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER
ner = NumpyAhocorasickNER()
ner.add_word("city", "Tokyo")
ner.fit()
ner.save("model.npz")

# ONNX backend — portable to any onnxruntime deployment
from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER
ner = OnnxAhocorasickNER()
ner.add_word("city", "Tokyo")
ner.fit()
ner.save("model")  # creates model.onnx + model.npz
```

See [`examples/`](examples/) for complete working examples and a benchmark script.

---

## Benchmarks

With 100k+ known phrases, this tool can tag documents in milliseconds thanks to the Aho-Corasick FSM structure. Run the benchmark yourself:

```bash
uv pip install ahocorasick-ner[numpy,onnx]
uv run python examples/benchmark.py
```

---

## Limitations

- Greedy longest-match only — no nested or overlapping entities
- No fuzzy matching (typos or misspellings won't match)
- All entities must be known beforehand

---

## License

Apache 2.0 — free for commercial and non-commercial use.

---

## Acknowledgements

- [pyahocorasick](https://github.com/WojciechMula/pyahocorasick) — C-based Aho-Corasick implementation
- [Hugging Face Datasets](https://huggingface.co/docs/datasets) — Domain-specific corpora
